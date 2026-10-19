# Script to evaluate a policy p in an environment e with the task t

# imports
import collections
import datetime
import json
import math
import os
import pathlib
import sys
import time
import traceback

import hydra
from hydra import compose
from omegaconf import DictConfig, OmegaConf

# Path setup so the isaac_sim.* imports resolve.
# File lives at isaac_sim/eval_policy.py; one parent hop -> repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Configs refer to the checkout as ${oc.env:INFRAVLA_ROOT}; default it to this repo.
os.environ.setdefault("INFRAVLA_ROOT", _REPO_ROOT)

from isaac_sim.task import Task
from isaac_sim.policies.policy import VLAPolicy as Policy
from isaac_sim.policies.waypoint_follower import WaypointFollower  # fine-grained follower


def _make_policy(run_cfg):
    return Policy(run_cfg.policy, run_cfg.controller, control_hz=run_cfg.control_hz)
from isaac_sim.sim.isaac_sim_env import IsaacSimEnv
from isaac_sim.sim.recorder import RunRecorder
from isaac_sim.utils.controller_utils import is_stuck  # shared with data_gen.py

from enum import Enum

# Stuck-detection knobs live in cfg.controller.stuck_*; see
# configs/controller/omnivla_default.yaml for the justifications.

class Action(Enum):
    RELOAD = 1
    KEEP_POLICY = 2
    KEEP_ENV = 3
    KEEP_BOTH = 4


# Per-run keys forwarded to the env config as `env.<key>=<value>` hydra
# overrides (spawn-pose probes: pin position/heading per run without one
# scene yaml per pose). Every key here recreates IsaacSimEnv, so all of them
# are rejected for KEEP_ENV/KEEP_BOTH — same fail-fast rule as env_scene/seed.
ENV_POSE_RUN_KEYS = (
    "start_xy",                  # [x, y] list in the run dict
    "start_heading_deg",
    "start_heading_mode",        # astar | fixed | random (see _common.py)
    "start_jitter_xy_m",         # 0.0 to disable a scene's baked-in jitter
    "start_jitter_heading_deg",
)


def _base_overrides_from_hydra() -> list:
    """The CLI/experiment overrides this app was launched with (e.g.
    `+experiment=...`), for re-application in per-run compose() calls.

    Without these, per-run configs silently revert to base-config defaults
    for everything the experiment set (recorder root, astar group, ...).
    Falls back to [] when no hydra app context exists (unit tests drive
    evaluate() via compose()).
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        return list(HydraConfig.get().overrides.task)
    except Exception:
        return []

def _write_summary_md(rows, dest):
    """Render the live sweep summary as Markdown. Rewritten in full after every
    row so Ctrl-C mid-sweep still leaves a usable artifact. Columns: env |
    task | seed | outcome | steps | final_dist | wall_s | run_dir."""
    lines = ["# Sweep summary", ""]
    lines.append(f"Total runs: {len(rows)}")
    by_outcome = {}
    for r in rows:
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
    lines.append("Outcomes: " + ", ".join(
        f"{k}={v}" for k, v in sorted(by_outcome.items())))
    lines.append("")
    lines.append("| env | task | seed | outcome | steps | final_dist (m) | wall_s | run_dir |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        rd = r.get("run_dir") or "—"
        rd_short = pathlib.Path(rd).name if rd != "—" else "—"
        fd = r.get("final_dist")
        # NaN is its own != → easiest way to detect it without depending on
        # math.isnan and handling string-/None-typed cases.
        fd_s = (f"{fd:.2f}" if isinstance(fd, (int, float)) and fd == fd
                else "—")
        seed_s = "—" if r.get("seed") is None else str(r["seed"])
        wall_s = r.get("wall_s")
        wall_s_str = (f"{wall_s:.1f}" if isinstance(wall_s, (int, float))
                      else "—")
        lines.append(
            f"| {r.get('env','?')} | {r.get('task','?')} | {seed_s} | "
            f"{r['outcome']} | {r.get('steps','—')} | {fd_s} | "
            f"{wall_s_str} | {rd_short} |"
        )
    dest.write_text("\n".join(lines) + "\n")


def evaluate(cfg):
    policy = None
    task = None
    env = None

    # Sweep-level summary dir: every `python eval_policy.py` invocation (even
    # a single-run one) writes a _sweep_<ts>/ subdir under recorder.runs_root:
    # summary.jsonl (one line per run) + summary.md (full table, rewritten
    # per-row).
    #
    # OmegaConf.select with a default lets the unit tests (which stub a
    # minimal cfg lacking the recorder block) reach the validators that
    # they're actually asserting on, rather than tripping on this lookup.
    sweep_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runs_root_str = OmegaConf.select(
        cfg, "recorder.runs_root", default="/tmp/infravla_eval_runs"
    )
    sweep_root = pathlib.Path(runs_root_str) / f"_sweep_{sweep_ts}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    summary_jsonl = sweep_root / "summary.jsonl"
    summary_md = sweep_root / "summary.md"
    rows: list[dict] = []
    sweep_t0 = time.time()
    print(f"[evaluate] sweep summary: {sweep_root}")
    print(f"[evaluate] {len(cfg.runs)} run(s) queued")

    #we have the pattern of one run config and then for each run we have the configs stored and use the override and hydra copose pattern to generate a new hydra config which will be given to the downstream functison and classes
    for run_idx, run in enumerate(cfg.runs):
        run_label = run.get("name", f"unnamed_{run_idx}")
        run_t0 = time.time()
        print(f"\n[evaluate] === run {run_idx + 1}/{len(cfg.runs)}: {run_label} ===")
        # Per-run try/except so a single transient failure (Kit hiccup, OOM,
        # asset-load timeout) doesn't abort the whole sweep — bad run gets
        # recorded with outcome='exception', sweep continues. *Config*
        # errors (ValueError from the override-vs-action validators below)
        # are programmer bugs, not transients, so they propagate up and
        # abort the whole job — same fail-fast semantics the unit tests
        # depend on.
        try:
            # Start from the app-level CLI/experiment overrides so per-run
            # compose() sees the same world the top-level cfg was built in;
            # per-run overrides come after, so they win on conflicts.
            overrides = _base_overrides_from_hydra()
            for key in ("policy", "task", "controller", "recorder"):
                if key in run:
                    #use the override
                    overrides.append(f"{key}={run[key]}")
            if "env_scene" in run:
                overrides.append(f"env/scene={run.env_scene}")
            # Spawn-pose probe keys -> env.* (see ENV_POSE_RUN_KEYS). `++` =
            # add-or-override: start_heading_mode is a dataclass default not
            # present in scene yamls, so a plain override would be rejected
            # by hydra ("key not in config").
            for key in ENV_POSE_RUN_KEYS:
                if key in run:
                    if key == "start_xy":
                        x, y = run.start_xy[0], run.start_xy[1]
                        overrides.append(f"++env.start_xy=[{float(x)},{float(y)}]")
                    else:
                        overrides.append(f"++env.{key}={run[key]}")
            if "lang_instruction" in run:
                # Per-run language instruction (lang eval sweeps: each scene
                # has its own target phrase, from the data-gen manifest).
                overrides.append(f'task.lang_instruction="{run.lang_instruction}"')  # double quotes — apostrophes ("witch's hat") break single-quoted Hydra overrides
            if "seed" in run:
                # Override the scene yaml's baked-in env.seed: re-roll
                # obstacle placement / bearing sampling without writing a
                # duplicate scene yaml per seed.
                overrides.append(f"env.seed={int(run.seed)}")
            if "fine_grained_enabled" in run:
                # Per-run toggle of the fine-grained (decoupled high-rate
                # follower) rollout loop. Affects ONLY the rollout control law,
                # never env/policy construction, so it composes freely with
                # KEEP_* actions (no validator needed) — e.g. an A/B that runs
                # the same scene with both rollout loops from one env boot.
                overrides.append(f"++fine_grained.enabled={bool(run.fine_grained_enabled)}")
            if "goal_graph_path" in run:
                # Per-run goal_graph_path override — required for parallel
                # sweeps. The default `task/image_graph.yaml` hardcodes a
                # single path; multiple processes running image_graph runs
                # in parallel would race on the same JSON + PNG outputs.
                # Each (env_scene, seed) should get a unique file. Pose tasks ignore the
                # field (env.set_task checks `if task.goal_graph_path:`),
                # so passing it for pose runs is harmless.
                overrides.append(f"task.goal_graph_path={run.goal_graph_path}")
            #use compose function and update eval (compose handles the recurisve pointers)
            run_cfg = compose(config_name="eval", overrides=overrides)
            #get action for potential overwriting
            action = Action[run.get("action", "RELOAD").upper()]
            if action in (Action.KEEP_POLICY, Action.KEEP_BOTH):
                for k in ("policy", "controller"):
                    if k in run:
                        raise ValueError(
                            f"run {run_label}: overrides '{k}' but action={action.name} "
                            f"won't recreate Policy — override would be silently dropped."
                        )
            if action in (Action.KEEP_ENV, Action.KEEP_BOTH):
                # Scene/seed/jitter NEED env reconstruction — reject. The
                # exact-pose keys are fine: they are applied to the kept env
                # via env.set_start_pose() below ("fast grid" repose path —
                # saves the ~8 min/run rebuild when only the spawn changes).
                # Jitter keys are config-time (template
                # __post_init__) so they can't re-roll on a kept env; an
                # explicit 0.0 is allowed (means "exact pose"), nonzero
                # would be silently ignored — reject it.
                for k in ("env_scene", "seed"):
                    if k in run:
                        raise ValueError(
                            f"run {run_label}: overrides {k!r} but action={action.name} "
                            f"won't recreate IsaacSimEnv — override would be silently dropped."
                        )
                for k in ("start_jitter_xy_m", "start_jitter_heading_deg"):
                    if k in run and float(run[k]) != 0.0:
                        raise ValueError(
                            f"run {run_label}: {k}={run[k]} with action={action.name} — "
                            f"jitter rolls at env construction and cannot re-roll on a "
                            f"kept env. Use exact start_xy/start_heading_deg, or RELOAD."
                        )
            task_kwargs = OmegaConf.to_container(run_cfg.task, resolve=True)
            if action == Action.RELOAD or policy is None or task is None or env is None:
                task = Task(**task_kwargs)
                if policy is not None:  # free the OLD 7B model BEFORE building the new one —
                    import gc, torch  # local import — torch only exists after Kit boot
                    del policy  # rebinding alone leaves 1-2 stale copies on the card
                    policy = None  # (23 GB grew to a 53 GB plateau; capped sweeps at 1-wide)
                    gc.collect()
                    torch.cuda.empty_cache()
                policy = _make_policy(run_cfg)
                env = IsaacSimEnv(run_cfg.env, record_hz=run_cfg.record_hz, physics_hz=run_cfg.physics_hz)
            elif action == Action.KEEP_POLICY:
                task = Task(**task_kwargs)
                env = IsaacSimEnv(run_cfg.env, record_hz=run_cfg.record_hz, physics_hz=run_cfg.physics_hz)
            elif action == Action.KEEP_ENV:
                task = Task(**task_kwargs)
                if policy is not None:  # same stale-copy free as the RELOAD branch
                    import gc, torch
                    del policy
                    policy = None
                    gc.collect()
                    torch.cuda.empty_cache()
                policy = _make_policy(run_cfg)
            elif action == Action.KEEP_BOTH:
                task = Task(**task_kwargs)
            else:
                raise RuntimeError("action doesn't exists")

            # Fast-grid repose: pose keys on a KEPT env re-aim the spawn
            # without reconstruction (validated above; evaluate_one_run's
            # env.reset() performs the actual placement). Must run BEFORE
            # env.set_task — image-graph recording depends on the new start.
            if action in (Action.KEEP_ENV, Action.KEEP_BOTH) and "start_xy" in run:
                env.set_start_pose(
                    (float(run.start_xy[0]), float(run.start_xy[1])),
                    start_heading_deg=(float(run.start_heading_deg)
                                       if "start_heading_deg" in run else None),
                    start_heading_mode=(str(run.start_heading_mode)
                                        if "start_heading_mode" in run else None),
                )

            #set node spacing before set_task
            task.set_node_spacing(run_cfg.controller.max_linear_vel_mps)
            # now set task ( goal graph will be calc )
            env.set_task(task)

            row = evaluate_one_run(policy, task, env, run_cfg, run_label)
        except ValueError:
            # Config-validation errors are programmer bugs — re-raise so the
            # caller (and the unit tests) sees them as a hard failure.
            raise
        except Exception as e:
            traceback.print_exc()
            row = {
                "name": run_label,
                "env": run.get("env_scene", "?"),
                "task": run.get("task", "?"),
                "seed": run.get("seed"),
                "outcome": "exception",
                "steps": None,
                "final_dist": None,
                "run_dir": None,
                "error": f"{type(e).__name__}: {e}",
            }
        row["wall_s"] = round(time.time() - run_t0, 1)
        rows.append(row)
        with open(summary_jsonl, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        _write_summary_md(rows, summary_md)
        print(f"[evaluate] run {run_idx + 1} done: outcome={row['outcome']} "
              f"steps={row.get('steps')} final_dist={row.get('final_dist')} "
              f"wall_s={row['wall_s']}")

    sweep_wall_s = round(time.time() - sweep_t0, 1)
    print(f"\n[evaluate] sweep complete: {len(rows)} runs in {sweep_wall_s}s "
          f"(={sweep_wall_s/3600:.2f} h). Summary: {sweep_root}")

def evaluate_one_run(policy, task, env, run_cfg, run_name):
    """Run one (env, task, policy) rollout. Returns a dict suitable for the
    sweep aggregator (env, task, seed, outcome, steps, final_dist, run_dir)."""
    controller_cfg = run_cfg.controller  # stuck-detection knobs live here

    obs = env.reset()
    # Eagerly resolved in IsaacSimEnv.__init__; just fetch the cached dict.
    effective_goal_pose = env.get_effective_goal_pose()
    policy.set_task(task, effective_goal_pose)
    policy.prime_blockade_signal(run_cfg.env.name)  # no-op unless policy.blockade_signal

    recorder = RunRecorder(run_cfg, run_name)

    # Hand the recorder the plain data it needs to draw the map (map + A* plan +
    # start/goal) — it never sees the live env. Done UNCONDITIONALLY (cheap, once
    # per run): the finalize rollout_viz.png is gated on path_viz.enabled, but the
    # per-step behaviour-panel map (save_robot_behavior, pose-mode) also needs it,
    # so we always supply it and let the recorder decide what to render.
    start_xy = env.template_cfg.start_xy
    goal_xy = effective_goal_pose["xy"]
    recorder.set_map(env.a_star.map, start_xy, goal_xy)
    spacing = float(run_cfg.viz.path_map.astar_sample_spacing_m)
    astar_path = env.get_a_start_path(start_xy, goal_xy).sample(spacing)
    recorder.set_astar_path(astar_path)

    # Rollout: legacy (one twist per ~3 Hz query, held 0.33 s) or fine-grained
    # (VLA queried at control_hz; a high-rate follower steers toward the predicted
    # lookahead waypoint (fixed wp4) in between — the momentum/spin fix). Opt-in via
    # cfg.fine_grained.enabled; default OFF leaves every existing eval config on
    # the legacy path. Both return (outcome, final_step, dist).
    fine_grained = bool(OmegaConf.select(run_cfg, "fine_grained.enabled", default=False))
    if fine_grained:
        outcome, final_step, dist = _rollout_fine_grained(
            policy, task, env, recorder, controller_cfg, effective_goal_pose, run_cfg, obs)
    else:
        outcome, final_step, dist = _rollout_legacy(
            policy, task, env, recorder, controller_cfg, effective_goal_pose, run_cfg, obs)

    # recorder.finalize renders <run_dir>/rollout_viz.png from the path it
    # accumulated + the map/A*-plan we handed it via set_map/set_astar_path
    # (gated on recorder.path_viz.enabled).
    recorder.finalize(outcome, dist)

    return {
        "name": run_name,
        "env": run_cfg.env.name,
        "task": run_cfg.task.name,
        "seed": int(run_cfg.env.seed) if run_cfg.env.get("seed") is not None else None,
        "outcome": outcome,
        # steps = number of policy queries actually issued. For reached/stuck,
        # final_step is the loop index at which we broke (0-based; +1 for a
        # count). For timeout, final_step stays at max_steps and we report
        # the full run length. (fine_grained reports the MODEL-QUERY count, not
        # the inner fast-tick count — same 3 Hz cadence as legacy.)
        "steps": (final_step + 1) if outcome != "timeout" else task.max_steps,
        "final_dist": float(dist),
        "run_dir": str(recorder.run_dir),
    }


def _rollout_legacy(policy, task, env, recorder, controller_cfg, effective_goal_pose, run_cfg, obs):
    """Legacy rollout: VLA queried once per control tick (~3 Hz), the resulting
    (v, w) held for one env.step (0.33 s). Returns (outcome, final_step, dist)."""
    stuck_window = int(controller_cfg.stuck_window)
    stuck_bbox_m = float(controller_cfg.stuck_bbox_m)
    stuck_grace = int(controller_cfg.stuck_grace_steps)
    recent_poses = collections.deque(maxlen=stuck_window)

    # No wall-clock pacing: sim time is synthetic — env.step
    # advances a fixed number of physics ticks per query, so the sim-time cadence
    # is exact regardless of wall clock. Sleeping to control_hz only matters on a
    # real robot; in sim it just wastes eval wall time.
    # control_period_s = 1.0 / float(run_cfg.control_hz)

    outcome = "timeout"
    # Default to max_steps so the timeout branch (loop exhausts) has a real
    # number to record. Reassigned to the actual breakpoint on early exit.
    final_step = task.max_steps
    dist = float("nan")
    for step in range(task.max_steps):
        # compute distance to goal inline — env has no get_robot_goal_dist method
        gx, gy = effective_goal_pose["xy"]
        dist = math.hypot(gx - obs["x"], gy - obs["y"])
        if dist < task.success_dist:
            outcome = "reached"
            final_step = step
            break
        recent_poses.append((obs["x"], obs["y"]))  # deque.append takes one item — wrap as a tuple
        if is_stuck(recent_poses, step, stuck_window, stuck_bbox_m, stuck_grace):
            outcome = "stuck"
            final_step = step
            break
        v, w, trace = policy(obs, step)
        recorder.record_step(step, obs, (v, w), dist)
        # Save the visualization BEFORE env.step so the rendered image
        # matches the (v, w) we're about to apply (same forward pass).
        recorder.save_robot_behavior(
            trace["cur_img"],
            trace["goal_img"],
            trace["goal_pose_loc_norm"],
            trace["waypoints"],
            v, w,
            policy.waypoint_spacing,
            trace["modality_id"],
            step,
        )
        obs = env.step((v, w))
        # Not relevant for sim (see pacing note above the loop) — only a real
        # robot needs the loop held to control_hz wall-clock.
        # elapsed = time.time() - t0
        # time.sleep(max(0, control_period_s - elapsed))
    return outcome, final_step, dist


def _rollout_fine_grained(policy, task, env, recorder, controller_cfg, effective_goal_pose, run_cfg, obs):
    """Fine-grained rollout — the momentum/spin fix.

    The VLA is queried at control_hz (~3 Hz) to predict a waypoint chunk; a
    WaypointFollower steers toward the FIXED lookahead waypoint (chunk index
    waypoint_select) at fine_grained.control_hz (30 Hz, mirroring data_gen.py),
    recomputing (v,w) from the live pose so heading commands update continuously
    instead of in 0.33 s stair-steps. Decouples the slow (~200 ms) model forward
    from the control rate without re-querying the model faster.

    Loop shape mirrors data_gen.py: each outer iteration advances ONE fast
    control tick; every `decimation`-th tick is a model-query + recording tick.
    decimation × steps_per_control == env.STEPS_PER_TICK, so the sim advances
    physically identically to the legacy loop — only the control law differs.
    Returns (outcome, final_step, dist), where final_step is the MODEL-QUERY
    count (3 Hz), matching the legacy step semantics.
    """
    stuck_window = int(controller_cfg.stuck_window)
    stuck_bbox_m = float(controller_cfg.stuck_bbox_m)
    stuck_grace = int(controller_cfg.stuck_grace_steps)
    recent_poses = collections.deque(maxlen=stuck_window)

    model_query_hz = float(run_cfg.control_hz)                       # VLA re-query rate (3 Hz)
    follower_hz = float(OmegaConf.select(run_cfg, "fine_grained.control_hz", default=30.0))
    physics_dt = 1.0 / float(run_cfg.physics_hz)
    steps_per_control = round(1.0 / (follower_hz * physics_dt))      # physics ticks per fast control tick
    decimation = round(follower_hz / model_query_hz)                # fast ticks per model query / recording

    follower = WaypointFollower(controller_cfg, model_query_hz=model_query_hz)

    outcome = "timeout"
    record_step = 0      # increments only on model-query (= recording) ticks
    final_step = task.max_steps
    dist = float("nan")
    trace = None
    total_control_steps = int(task.max_steps) * decimation
    for control_step in range(total_control_steps):
        # Goal check — every fast tick, so a successful approach terminates
        # promptly rather than on the next 3 Hz boundary.
        gx, gy = effective_goal_pose["xy"]
        dist = math.hypot(gx - obs["x"], gy - obs["y"])
        if dist < task.success_dist:
            outcome = "reached"
            final_step = record_step
            break

        is_query_tick = (control_step % decimation == 0)
        if is_query_tick:
            # Stuck detection is defined over the recorded (3 Hz) pose history.
            recent_poses.append((obs["x"], obs["y"]))
            if is_stuck(recent_poses, record_step, stuck_window, stuck_bbox_m, stuck_grace):
                outcome = "stuck"
                final_step = record_step
                break
            # MODEL QUERY (~200 ms): predict the lookahead waypoint (world frame)
            # and install it as the follower's carrot. obs has images here
            # (capture_next set it on the previous fast tick / reset()).
            target_xy, trace = policy.predict_target_waypoint(obs)
            follower.set_target(target_xy)

        # High-rate steering toward the fixed carrot — recomputed from the live
        # pose EVERY fast tick (this recompute is the actual spin fix).
        v, w = follower(obs)

        if is_query_tick:
            recorder.record_step(record_step, obs, (v, w), dist)
            # Save viz BEFORE env.step so the rendered image matches (v, w).
            recorder.save_robot_behavior(
                trace["cur_img"],
                trace["goal_img"],
                trace["goal_pose_loc_norm"],
                trace["waypoints"],
                v, w,
                policy.waypoint_spacing,
                trace["modality_id"],
                record_step,
            )
            record_step += 1

        # Capture images only when the NEXT fast tick will model-query (the obs
        # consumed at query tick T comes from this env.step at T-1). Non-query
        # ticks skip the 5-camera readback — the follower reads pose only.
        capture_next = ((control_step + 1) % decimation == 0)
        obs = env.step((v, w), n_physics_ticks=steps_per_control, capture_images=capture_next)
    return outcome, final_step, dist

@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    evaluate(cfg)


if __name__ == "__main__":
    main()
