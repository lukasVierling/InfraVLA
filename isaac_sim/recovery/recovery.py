import json
import math
import os
import pathlib
import re
import shutil
import sys
import time

import numpy as np

import hydra
from omegaconf import OmegaConf, open_dict

_REPO_ROOT = os.path.abspath(next(
    p for p in pathlib.Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists()))  # depth-independent (CLEANUP.md R9)
for _p in (_REPO_ROOT, os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Configs refer to the checkout as ${oc.env:INFRAVLA_ROOT}; default it to this repo.
os.environ.setdefault("INFRAVLA_ROOT", _REPO_ROOT)

from isaac_sim.utils.recovery_analysis import (find_derailment,
                                               find_excursions,
                                               project_to_gt_indices)
from isaac_sim.utils.cf_pairing import load_pose_stream


def seal_of(scene_name: str) -> str:
    """Seal pattern from a cf run-dir name: warehouse_2x3_1721_cfBE__... -> 'BE'.

    Recorded per mined anchor so the set's composition (how many cfBE, cfF, cf0
    ...) is readable from the manifest, and so an unsealed-scene ablation needs
    no re-mining. '' when the name carries no _cf tag.
    """
    m = re.search(r"_cf([A-Z0]+)__", scene_name)
    return m.group(1) if m else ""


def _snippet_end(window_steps: set, t: int) -> int:
    """return the end of the snippet"""
    hi = t
    while (hi + 1) in window_steps:
        hi += 1
    return hi


def _model_rollout(policy, env, run_cfg, task_kwargs, spawn_xy, spawn_yaw,
                   budget_steps, run_name, astar_path=None):
    """do a model rollout given a polcy and an env"""
    from isaac_sim.task import Task
    from isaac_sim.eval_policy import _rollout_legacy, _rollout_fine_grained
    from isaac_sim.sim.recorder import DataGenRecorder

    #new recovorder that doesn't save any visualizations
    class _NoVizRecorder(DataGenRecorder):
        def save_robot_behavior(self, *args, **kwargs):
            pass
    #set the start pose of the env provided thorugh arguments
    env.set_start_pose(tuple(spawn_xy),
                       start_heading_deg=math.degrees(spawn_yaw),
                       start_heading_mode="fixed")
    #reset the environment
    obs = env.reset()
    #colculated the effective goal pose
    effective_goal_pose = env.get_effective_goal_pose()
    #get the task for hte robot
    task = Task(**{**task_kwargs, "max_steps": int(budget_steps)})
    #set the task of the policy
    policy.set_task(task, effective_goal_pose)
    #idk what this is? 
    policy.prime_blockade_signal(run_cfg.env.name)

    #instantiate the reocrder
    recorder = _NoVizRecorder(run_cfg, run_name)
    recorder.set_map(env.a_star.map, tuple(spawn_xy),
                     tuple(effective_goal_pose["xy"]))
    if astar_path is not None:
        recorder.set_astar_path([(float(q[0]), float(q[1])) for q in astar_path])
    #find out whether fine grained control frequency
    fine_grained = bool(OmegaConf.select(run_cfg, "fine_grained.enabled",
                                         default=False))
    #get a rollout function
    rollout = _rollout_fine_grained if fine_grained else _rollout_legacy
    #do the rollout ge the result and the final step which we don't use here
    outcome, _final_step, dist = rollout(
        policy, task, env, recorder, run_cfg.controller, effective_goal_pose,
        run_cfg, obs)
    #wrap up recording 
    recorder.finalize(outcome, dist)
    #return the run directory where the run is saved and the outcome of the run
    return recorder.run_dir, outcome


def _oracle_rollout(env, run_cfg, task_kwargs, spawn_xy, spawn_yaw, goal_xy,
                    run_name, path_points):
    """does a policy rollout from a givne startpoint to a given endpoint,
    following the GIVEN path (the original route, trimmed at the spawn) —
    never a fresh A* plan (wrong-turn lock-in risk, Lukas 2026-07-17)."""
    from isaac_sim.policies.a_star_policy import A_star_policy
    from isaac_sim.sim.recorder import DataGenRecorder
    from isaac_sim.task import Task
    #create an oracle rollout where the oracle is the GT so we have no policy
    rcfg = run_cfg.recovery
    #also set the task
    task = Task(**{**task_kwargs, "max_steps": int(rcfg.oracle_max_record_steps)})
    #set the starting pose
    env.set_start_pose(tuple(spawn_xy),
                       start_heading_deg=math.degrees(spawn_yaw),
                       start_heading_mode="fixed")
    #reset the env
    obs = env.reset()
    #use the provided (original-route) path — no replanning here.
    #plain python floats: numpy scalars break _sign() in solve_twist.
    path = [(float(p[0]), float(p[1])) for p in path_points]
    #instantiate the oracle as a star policy
    oracle = A_star_policy(run_cfg.policy_oracle, run_cfg.controller,
                           control_hz=rcfg.oracle_control_hz)
    #set the path of the oracle
    oracle.set_path(path)

    #start the data recorder
    recorder = DataGenRecorder(run_cfg, run_name)
    recorder.set_map(env.a_star.map, tuple(spawn_xy), tuple(goal_xy))
    #set the a start path for the data recorder, why does hte recorder need the a star path
    recorder.set_astar_path(path)

    #some params
    control_hz = float(rcfg.oracle_control_hz)
    physics_dt = 1.0 / float(run_cfg.physics_hz)
    steps_per_control = round(1.0 / (control_hz * physics_dt))
    decimation = round(control_hz / float(run_cfg.record_hz))

    #whole control loop write up with oracle
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    outcome = "timeout"
    dist = float("inf")
    record_step = 0
    for control_step in range(int(task.max_steps) * decimation):
        dist = math.hypot(gx - obs["x"], gy - obs["y"])
        if dist < task.success_dist:
            outcome = "reached"
            break
        v, w, _ = oracle(obs, control_step)
        if control_step % decimation == 0:
            recorder.record_step(record_step, obs, (v, w), dist)
            record_step += 1
        capture_next = ((control_step + 1) % decimation == 0)
        obs = env.step((v, w), n_physics_ticks=steps_per_control,
                       capture_images=capture_next)
    #wrap up after run
    recorder.finalize(outcome, dist)
    return recorder.run_dir, outcome


def _stitch(failure_dir, recovery_dir, out_dir, cut_idx):
    """function that takes a failure and recovery dir and an idx and stiteches them together"""
    from isaac_sim.planning.a_star.goal_graph import GoalGraph, GraphNode
    #get the fail and reocvery data
    g_fail = GoalGraph.load(failure_dir / "frames.json")
    g_rec = GoalGraph.load(recovery_dir / "frames.json")

    os.makedirs(out_dir, exist_ok=True)
    cam_dirs = sorted(d.name for d in os.scandir(recovery_dir)
                      if d.is_dir() and any(f.name.endswith(".png")
                                            for f in os.scandir(d.path)))
    for cam in cam_dirs:
        os.makedirs(out_dir / cam, exist_ok=True)

    sources = [(failure_dir, n) for n in g_fail.nodes[:cut_idx]] + \
              [(recovery_dir, n) for n in g_rec.nodes]
    nodes = []
    for i, (src_dir, node) in enumerate(sources):
        new_name = f"{i:03d}.png"
        for cam in cam_dirs:
            shutil.copy2(src_dir / cam / node.image_filename,
                         out_dir / cam / new_name)
        nodes.append(GraphNode(xy=node.xy, yaw_rad=node.yaw_rad,
                               image_filename=new_name,
                               cmd_v=node.cmd_v, cmd_w=node.cmd_w))

    GoalGraph(env_config_name=g_rec.env_config_name,
              env_config_hash=g_rec.env_config_hash,
              node_spacing_m=0.0, robot_inflation_m=g_rec.robot_inflation_m,
              sampling_mode="recovery_stitched",
              target_speed_mps=g_rec.target_speed_mps,
              sample_hz=g_rec.sample_hz,
              nodes=nodes).save(out_dir / "frames.json")

    rec_result = json.loads((recovery_dir / "result.json").read_text())
    (out_dir / "result.json").write_text(json.dumps({
        "outcome": "reached",
        "steps": len(nodes),
        "final_dist": rec_result["final_dist"],
        "n_frames": len(nodes),
        "sample_hz": rec_result["sample_hz"],
        "astar_path_length_m": rec_result["astar_path_length_m"],
        "stitched": {"n_failure_frames": cut_idx,
                     "n_recovery_frames": len(g_rec.nodes)},
    }, indent=2))
    shutil.copy2(recovery_dir / "astar_path.json", out_dir / "astar_path.json")
    shutil.copy2(recovery_dir / "config.yaml", out_dir / "config.yaml")
    return len(nodes)


# config_path is relative to THIS file; configs/ sits one level up since the
# recovery scripts moved into isaac_sim/recovery/ (2026-09-07).
@hydra.main(version_base=None, config_path="../configs", config_name="recovery")
def main(cfg):
    from prismatic.vla.datasets.dataset_utils import mine_cf_windows
    from isaac_sim.sim.isaac_sim_env import IsaacSimEnv
    from isaac_sim.eval_policy import _make_policy
    from isaac_sim.task import Task
    from pathlib import Path

    rcfg = cfg.recovery
    split_dir = Path(rcfg.split_dir)
    output_root = Path(rcfg.output_root)
    shard, n_shards = int(rcfg.shard), int(rcfg.n_shards)
    # Per-shard staging: a sibling shard's end-of-run cleanup must not touch ours.
    staging_root = output_root / f"_staging_shard{shard}"
    os.makedirs(staging_root, exist_ok=True)
    rng = np.random.default_rng(int(rcfg.seed))

    # ---- The cf dataset's samples, taken as given: same mining function the
    # dataset machinery uses (one window definition everywhere). A "sample"
    # is (traj, t) with t inside a decision window; its snippet ends at the
    # window's last step.
    traj_dirs = sorted((d for d in split_dir.iterdir() if d.is_dir()),
                       key=lambda d: d.name)
    names = [d.name for d in traj_dirs]
    streams = [load_pose_stream(d) for d in traj_dirs]
    mining = mine_cf_windows(
        traj_names=names,
        traj_positions=[s[0] for s in streams],
        n_chunk=int(rcfg.n_chunk),
        dev_thresh_m=float(rcfg.dev_thresh_m),
        pre_steps=int(rcfg.window_pre_steps),
        post_steps=int(rcfg.window_post_steps),
        criterion=str(rcfg.criterion),
        traj_yaws=[s[1] for s in streams],
        yaw_eps_deg=float(rcfg.yaw_eps_deg))
    # Replay mode (Lukas 2026-07-17): instead of the random sample stream,
    # roll exactly the DERAILED (scene, t) records of an earlier campaign's
    # manifest — "did the fine-tune fix the known failure states?".
    if rcfg.replay_manifest:
        man = json.loads(Path(rcfg.replay_manifest).read_text())
        fails = [r for r in man["records"] if r.get("derailed")]
        name_to_ti = {d.name: i for i, d in enumerate(traj_dirs)}
        # dedupe: n_spawns > 1 writes one record per spawn of the SAME state
        seen = set()
        samples = []
        for r in fails:
            key = (name_to_ti[r["scene"]], int(r["t"]))
            if key not in seen:
                seen.add(key)
                samples.append(key)
        order = list(range(len(samples)))[shard::n_shards]
        print(f"[recovery] REPLAY: {len(samples)} failure states from "
              f"{rcfg.replay_manifest}; shard {shard}/{n_shards} -> "
              f"{len(order)} samples; measure_only={rcfg.measure_only}")
    elif str(OmegaConf.select(cfg, "recovery.anchor_mode", default="random")) == "spawn":
        samples = [(ti, 0) for ti, steps in mining["windows_by_traj"].items()
                   if steps]
        order = rng.permutation(len(samples))[shard::n_shards]
        print(f"[recovery] ANCHOR=spawn: {len(samples)} trajectories (one FULL "
              f"rollout each from the scene's own spawn); shard "
              f"{shard}/{n_shards} -> {len(order)} samples; "
              f"quota={rcfg.quota} seed={rcfg.seed}")
    elif str(OmegaConf.select(cfg, "recovery.anchor_mode", default="random")) == "window_start":
        samples = [(ti, min(steps)) for ti, steps in mining["windows_by_traj"].items()
                   if steps]
        order = rng.permutation(len(samples))[shard::n_shards]
        print(f"[recovery] ANCHOR=window_start: {len(samples)} trajectories "
              f"(one probe each, earliest frame of the cf window); shard "
              f"{shard}/{n_shards} -> {len(order)} samples; "
              f"quota={rcfg.quota} seed={rcfg.seed}")
    else:
        samples = [(ti, t) for ti, steps in mining["windows_by_traj"].items()
                   for t in sorted(steps)]
        order = rng.permutation(len(samples))[shard::n_shards]
        print(f"[recovery] {len(samples)} cf samples across "
              f"{sum(1 for s in mining['windows_by_traj'].values() if s)} "
              f"trajectories; shard {shard}/{n_shards} -> {len(order)} samples; "
              f"quota={rcfg.quota} seed={rcfg.seed}")

    policy = None
    records = []
    n_collected = 0
    n_rollouts = 0
    t0 = time.time()

    for si in order:
        if n_collected >= int(rcfg.quota) or n_rollouts >= int(rcfg.max_rollouts):
            break
        ti, t = samples[si]
        src_dir = traj_dirs[ti]
        gt_pos, gt_yaws, _ = streams[ti][0], streams[ti][1], None
        hi = _snippet_end(mining["windows_by_traj"][ti], t)

        # Fresh env per sample — no cross-sample reuse (see notes).
        src_cfg = OmegaConf.load(src_dir / "config.yaml")
        run_cfg = OmegaConf.merge(cfg)
        with open_dict(run_cfg):
            run_cfg.env = src_cfg.env
        run_cfg.recorder.runs_root = str(staging_root)
        task_kwargs = OmegaConf.to_container(run_cfg.task, resolve=True)
        env = IsaacSimEnv(run_cfg.env, record_hz=run_cfg.record_hz,
                          physics_hz=run_cfg.physics_hz)
        env.set_task(Task(**task_kwargs))
        if policy is None:
            policy = _make_policy(run_cfg)

        snippet_len = hi - t + 1
        spawn_mode = (str(OmegaConf.select(cfg, "recovery.anchor_mode",
                                           default="random")) == "spawn")
        min_peak = float(OmegaConf.select(cfg, "recovery.min_excursion_peak_m",
                                          default=0.0))
        if spawn_mode:
            # full-episode probe: the budget must reach the CONSEQUENCE of the
            # decision (the wedge sits ~40-60 steps past the junction), so it
            # scales with the whole route, not the window remainder.
            budget = int(round(float(rcfg.budget_factor) * len(gt_pos)))
        else:
            budget = int(round(float(rcfg.budget_factor)
                               * max(snippet_len, int(rcfg.min_rollout_steps))))
        tag = f"{src_dir.name.split('__')[0]}_t{t}"
        src_plan = np.array(json.loads(
            (src_dir / "astar_path.json").read_text())["path_xy"])
        fail_dir, roll_outcome = _model_rollout(
            policy, env, run_cfg, task_kwargs,
            spawn_xy=gt_pos[t], spawn_yaw=float(gt_yaws[t]),
            budget_steps=budget, run_name=f"fail_{tag}",
            astar_path=src_plan)
        n_rollouts += 1

        roll_pos, roll_yaws, _ = load_pose_stream(fail_dir)
        res = find_derailment(
            roll_pos, gt_pos[t:],
            dev_thresh_m=float(rcfg.rollout_dev_thresh_m),
            sustain_steps=int(rcfg.sustain_steps),
            aligned_eps_m=float(rcfg.aligned_eps_m))
        if spawn_mode:
            # ALL excursions (rejoin semantics), each projected onto GT frames
            # and kept only if its spawn band overlaps the cf window — the
            # in-window filter that keeps spawn blips out of the dataset while
            # still RECORDING them (measurable, not invisible).
            win = mining["windows_by_traj"][ti]
            wlo, whi = min(win), max(win)
            js = project_to_gt_indices(roll_pos, gt_pos)
            exc = find_excursions(roll_pos, gt_pos,
                                  dev_thresh_m=float(rcfg.rollout_dev_thresh_m),
                                  aligned_eps_m=float(rcfg.aligned_eps_m))
            for e in exc:
                e["gt_band"] = [int(js[e["last_aligned_idx"]]),
                                int(js[e["onset_idx"]])]
                # RECORDED, not gating (2026-08-24): the cf window is anchored
                # on the twins' FIRST divergence, which for 65% of this split
                # is the spawn itself — it cannot see the zone-D south-aisle
                # decision where the policy actually fails. Kept in the
                # manifest so the window-vs-severity choice stays auditable.
                e["in_window"] = not (e["gt_band"][1] < wlo
                                      or e["gt_band"][0] > whi)
            bands = [(e["last_aligned_idx"], e["onset_idx"])
                     for e in exc if e["peak_dev_m"] >= min_peak]
            rec = dict(scene=src_dir.name, t=int(t), snippet_end=int(hi),
                       seal=seal_of(src_dir.name),
                       rollout_outcome=roll_outcome,
                       derailed=bool(bands),
                       max_dev_m=res["max_dev_m"], n_steps=len(roll_pos),
                       excursions=exc, window=[int(wlo), int(whi)])
        else:
            bands = ([(res["last_aligned_idx"], res["onset_idx"])]
                     if res["derailed"] else [])
            rec = dict(scene=src_dir.name, t=int(t), snippet_end=int(hi),
                       rollout_outcome=roll_outcome,
                       derailed=bool(res["derailed"]),
                       max_dev_m=res["max_dev_m"], n_steps=len(roll_pos))
        if spawn_mode:
            peaks = ", ".join(f"{e['peak_dev_m']:.1f}"
                              f"{'' if e['rejoined'] else '!'}" for e in exc)
            print(f"[recovery] {tag}: excursions={len(exc)} [{peaks}] "
                  f"severe(>={min_peak}m)={len(bands)} "
                  f"({n_collected}/{rcfg.quota} collected, {n_rollouts} rollouts)")
        else:
            print(f"[recovery] {tag}: derailed={res['derailed']} "
                  f"max_dev={res['max_dev_m']:.2f}m "
                  f"({n_collected}/{rcfg.quota} collected, {n_rollouts} rollouts)")

        if rcfg.measure_only:
            # Replay measurement: verdict only — no oracle, no stitching.
            # keep_rollouts: archive the full rollout recording (poses+images)
            # under output_root/rollouts/ for visual failure analysis.
            if rcfg.keep_rollouts:
                keep_dir = output_root / "rollouts" / f"replay_{tag}"
                keep_dir.parent.mkdir(exist_ok=True)
                shutil.move(str(fail_dir), str(keep_dir))
            else:
                shutil.rmtree(fail_dir)
            records.append(rec)
            continue

        if not bands:
            shutil.rmtree(fail_dir)
            records.append(rec)
            continue

        # Spawns: n_spawns random points between last-aligned and divergence
        # onset, DISTINCT per derailment (Lukas 2026-07-17: the band IS the
        # hard initiate-the-correction zone — more draws there = more of the
        # scarce transition supervision; capped by the band's width).
        # Recovery = follow the ORIGINAL route (source astar_path trimmed at
        # the spawn-nearest node) to the ORIGINAL goal — never a fresh A*
        # plan, which a wrong-aisle-leaning spawn could tip into locking in
        # the failure (Lukas 2026-07-17). Non-reaching runs are filtered.
        src_path = np.array(json.loads(
            (src_dir / "astar_path.json").read_text())["path_xy"])
        spawn_draws = []          # (band_idx, spawn_idx, band_onset)
        for b, (b_la, b_on) in enumerate(bands):
            band = list(range(b_la, b_on + 1))
            n_spawns = min(int(rcfg.n_spawns), len(band))
            for s in rng.choice(band, size=n_spawns, replace=False):
                spawn_draws.append((b, int(s), int(b_on)))
        for k, (b, spawn_idx, band_onset) in enumerate(spawn_draws):
            spawn_xy = roll_pos[spawn_idx]
            j = int(np.linalg.norm(src_path - spawn_xy, axis=1).argmin())
            path_points = src_path[j:]
            goal_xy = path_points[-1]
            rec_dir, rec_outcome = _oracle_rollout(
                env, run_cfg, task_kwargs,
                spawn_xy=spawn_xy,
                spawn_yaw=float(roll_yaws[spawn_idx]),
                goal_xy=goal_xy, run_name=f"rec_{tag}_s{k}",
                path_points=path_points)

            rec_k = dict(rec, spawn=k, spawn_idx=spawn_idx, band=b,
                         recovery_outcome=rec_outcome)
            if rec_outcome == "reached":
                # legacy name for spawn 0; _s<k> suffix for extras (t stays
                # parseable — consumers use recovery_analysis.parse_rec_dirname).
                # Spawn mode: t would be 0 for every band -> use the band's
                # ONSET rollout index as the unique, meaningful t.
                t_name = band_onset if spawn_mode else t
                suffix = f"_t{t_name}" if k == 0 else f"_t{t_name}_s{k}"
                out_dir = output_root / f"rec_{src_dir.name}{suffix}"
                n_frames = _stitch(fail_dir, rec_dir, out_dir,
                                   cut_idx=spawn_idx)
                rec_k["stitched_frames"] = n_frames
                n_collected += 1
                print(f"[recovery]   STITCHED {out_dir.name}: {spawn_idx} "
                      f"failure + {n_frames - spawn_idx} recovery frames")
            shutil.rmtree(rec_dir)
            records.append(rec_k)
        shutil.rmtree(fail_dir)

    shutil.rmtree(staging_root, ignore_errors=True)
    manifest = dict(
        checkpoint=OmegaConf.to_container(cfg.policy, resolve=False),
        split_dir=str(split_dir),
        quota=int(rcfg.quota),
        n_collected=n_collected,
        n_failures=sum(1 for r in records if r["derailed"]),
        n_rollouts=n_rollouts,
        knobs=OmegaConf.to_container(rcfg, resolve=True),
        records=records,
        elapsed_s=round(time.time() - t0, 1),
    )
    manifest_name = ("recovery_manifest.json" if n_shards == 1
                     else f"recovery_manifest_shard{shard}.json")
    (output_root / manifest_name).write_text(json.dumps(manifest, indent=2))
    print(f"[recovery] DONE: {n_collected}/{rcfg.quota} stitched from "
          f"{manifest['n_failures']} failures / {n_rollouts} rollouts in "
          f"{manifest['elapsed_s']}s -> {output_root}")


if __name__ == "__main__":
    main()
