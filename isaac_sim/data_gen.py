# Script to generate training data by rolling out an A* oracle policy in the env.
# Mirrors eval_policy.py's structure — same run-loop, recorder, stuck detection —
# but swaps the VLA policy for an A* path-following oracle.

import collections
import math
import os
import sys

import hydra
from hydra import compose
from hydra.core.hydra_config import HydraConfig  # read CLI overrides for per-run re-compose
from omegaconf import OmegaConf

# Path setup so `isaac_sim.*` package imports resolve
# regardless of cwd. File lives at isaac_sim/data_gen.py;
# one parent hop lands on the repo root (which contains
# the `isaac_sim` package).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Configs refer to the checkout as ${oc.env:INFRAVLA_ROOT}; default it to this repo.
os.environ.setdefault("INFRAVLA_ROOT", _REPO_ROOT)

from isaac_sim.task import Task
from isaac_sim.sim.isaac_sim_env import IsaacSimEnv
from isaac_sim.sim.recorder import DataGenRecorder  # subclass: emits frames.json + astar_path.json
from isaac_sim.utils.controller_utils import is_stuck  # shared with eval_policy.py

from isaac_sim.policies.a_star_policy import A_star_policy as Policy
from isaac_sim.planning.a_star.planner import PlanningError  # per-run skip on sampler/USD edge cases
from isaac_sim.utils.planning_viz import save_planning_viz  # per-run SDF + A* + headings PNG

from enum import Enum


class Action(Enum):
    RELOAD = 1
    KEEP_ENV = 2


def generate(cfg):
    try:
        cli_overrides = list(HydraConfig.get().overrides.task)
    except Exception:
        cli_overrides = []  # unit tests run generate() without HydraConfig set

    task = None
    env = None
    policy = None
    for run in cfg.runs:
        overrides = list(cli_overrides)
        # `controller` is here too — the A* policy reads controller.dt_s,
        # eps, max_*; without this override key, run-level controller knobs
        # would be silently dropped (parallel to eval_policy.py).
        for key in ("policy", "task", "controller", "recorder"):
            if key in run:
                overrides.append(f"{key}={run[key]}")
        if "env_scene" in run:
            overrides.append(f"env/scene={run.env_scene}")
        if "seed" in run:
            overrides.append(f"env.seed={int(run.seed)}")
        if "goal_graph_path" in run:
            overrides.append(f"task.goal_graph_path={run.goal_graph_path}")
        run_cfg = compose(config_name="data_gen", overrides=overrides)
        action = Action[run.get("action", "RELOAD").upper()]
        if action == Action.KEEP_ENV:
            for k in ("env_scene", "seed"):
                if k in run:
                    raise ValueError(
                        f"run {run.get('name', '?')}: overrides {k!r} but action=KEEP_ENV "
                        f"won't recreate IsaacSimEnv — override would be silently dropped."
                    )
        task_kwargs = OmegaConf.to_container(run_cfg.task, resolve=True)
        try:
            if action == Action.RELOAD or task is None or env is None or policy is None:
                task = Task(**task_kwargs)
                env = IsaacSimEnv(run_cfg.env, record_hz=run_cfg.record_hz, physics_hz=run_cfg.physics_hz)
                policy = Policy(run_cfg.policy, run_cfg.controller, control_hz=run_cfg.control_hz)
            elif action == Action.KEEP_ENV:
                task = Task(**task_kwargs)
                policy = Policy(run_cfg.policy, run_cfg.controller, control_hz=run_cfg.control_hz)
            else:
                raise RuntimeError(f"unknown action: {action}")
            env.set_task(task)  # parallel to eval_policy.py — env needs the task before reset
            rollout(policy, task, env, run_cfg, run.get("name", "unnamed"))
        except PlanningError as e:
            print(f"[data_gen] SKIP run {run.get('name', '?')}: PlanningError — {e}")
            # If the env partially constructed before raising, drop it so the
            # NEXT run takes the RELOAD path and builds a fresh env. Otherwise
            # a KEEP_ENV next run would reuse a half-constructed env.
            task, env, policy = None, None, None


def rollout(policy, task, env, run_cfg, run_name):
    controller_cfg = run_cfg.controller
    # Rate knobs are top-level cfg fields (see data_gen.yaml).
    record_hz = float(run_cfg.record_hz)
    control_hz = float(run_cfg.control_hz)
    physics_dt = 1.0 / float(run_cfg.physics_hz)  # inner PhysX dt; physics_hz is the single-source cfg knob (see data_gen.yaml)
    steps_per_control = round(1.0 / (control_hz * physics_dt))
    decimation = round(control_hz / record_hz)  # control ticks per recording tick
    obs = env.reset()
    effective_goal_pose = env.get_effective_goal_pose()
    # Plan from the template start to the EFFECTIVE goal (= standoff-shifted),
    # then densify into an arc-length-uniform (x, y) list for pure-pursuit
    # lookahead. The policy and recorder both consume a flat list of waypoints,
    # so we materialise the samples here once and hand the same list to both.
    planned_path = env.get_a_start_path(
        env.template_cfg.start_xy,
        effective_goal_pose["xy"],
    )
    path = planned_path.sample(float(run_cfg.policy.densify_spacing_m))

    policy.set_path(path)

    recorder = DataGenRecorder(run_cfg, run_name)  # writes frames.json + astar_path.json on finalize
    recorder.set_astar_path(path)  # persisted to astar_path.json at finalize

    # One-shot diagnostic: SDF map + planned A* path + per-node heading arrows.
    # Cheap (planning re-runs internally — ms scale) and gives a per-run sanity
    # check that the planner did what we expect on this specific scene.
    save_planning_viz(env, recorder.run_dir / "planning_viz.png", run_cfg.viz.path_map)

    stuck_window = int(controller_cfg.stuck_window)
    stuck_bbox_m = float(controller_cfg.stuck_bbox_m)
    stuck_grace = int(controller_cfg.stuck_grace_steps)
    recent_poses = collections.deque(maxlen=stuck_window)

   
    outcome = "timeout"
    dist = float("inf")  # so recorder.finalize gets a defined value if max_steps == 0
    record_step = 0      # increments only on recording ticks; passed to recorder + is_stuck
    total_control_steps = int(task.max_steps) * decimation

    for control_step in range(total_control_steps):
        # Goal check — runs every fast tick so a successful approach terminates
        # promptly, not on the next 3 Hz boundary.
        gx, gy = effective_goal_pose["xy"]
        dist = math.hypot(gx - obs["x"], gy - obs["y"])
        if dist < task.success_dist:
            outcome = "reached"
            break

        is_recording_tick = (control_step % decimation == 0)

        # Stuck detection is defined over the *recorded* pose history (3 Hz
        # samples), so only update the buffer and check on recording ticks.
        if is_recording_tick:
            recent_poses.append((obs["x"], obs["y"]))  # deque.append takes one item — wrap as a tuple
            if is_stuck(recent_poses, record_step, stuck_window, stuck_bbox_m, stuck_grace):
                outcome = "stuck"
                break

        # Policy query — every control tick. The A* oracle's PD controller in
        # a_star_policy.py uses controller.dt_s = 1/control_hz, so (v, w) is
        # scaled for "reach the carrot in one fast tick", not one recording tick.
        v, w, _ = policy(obs, control_step)  # 3rd return is a viz trace; oracle emits {} so just drop

        if is_recording_tick:
            # record_step (recorder.RunRecorder.record_step) writes the current obs
            # + the command we're about to apply. Image basename matches record_step,
            # so frames.json's image_filename references resolve correctly.
            recorder.record_step(record_step, obs, (v, w), dist)
            record_step += 1

        # Advance sim by one fast control tick (1/control_hz s). Default
        # n_physics_ticks=env.STEPS_PER_TICK would advance 1/record_hz instead
        # — we explicitly pass steps_per_control here.
        # capture_images only when the NEXT tick will record this obs (recorded
        # obs at tick T comes from this env.step at T-1). Non-recording ticks
        # skip the 5-camera readback — the oracle reads pose only.
        capture_next = ((control_step + 1) % decimation == 0)
        obs = env.step((v, w), n_physics_ticks=steps_per_control, capture_images=capture_next)

    recorder.finalize(outcome, dist)  # finalize signature: (outcome, final_dist); n_steps tracked internally


@hydra.main(version_base=None, config_path="configs", config_name="data_gen")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    generate(cfg)


if __name__ == "__main__":
    main()
