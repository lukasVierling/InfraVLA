import datetime as dt
import hashlib
import json
import math
import pathlib
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

# GoalGraph / GraphNode + planning_viz are imported inside methods, not at
# module top, so the recorder imports without Isaac Sim. See notes at EOF.


class RunRecorder:
    """Eval-side recorder: per-step jsonl log + per-cam images + result.json.

    Image-naming and metadata-emission are exposed as small hooks
    (_image_filename, _write_cam_images, _result_extras) so DataGenRecorder
    can override naming + add GoalGraph emission without re-implementing
    __init__ or the log/result plumbing.
    """

    def __init__(self, run_cfg, run_name):
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = pathlib.Path(run_cfg.recorder.runs_root) / f"{run_name}__{run_cfg.task.name}__{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_cfg = run_cfg
        _log_frames = run_cfg.recorder.get("log_frames", True)
        self.cam_dirs = {
            cam_name: self.run_dir / cam_name
            for cam_name, cam in run_cfg.env.cameras.items()
            if _log_frames and getattr(cam, "logging", False)
        }
        for d in self.cam_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        print(f"[RunRecorder] logging frames for cameras: {sorted(self.cam_dirs.keys())}")

        # Dump the FULL composed run config so the run is replayable from one file.
        OmegaConf.save(config=run_cfg, f=str(self.run_dir / "config.yaml"))

        self.log_file = open(self.run_dir / "log.jsonl", "w", buffering=1)
        self.n_steps = 0
        print(f"[RunRecorder] run dir: {self.run_dir}")

        # Executed (x, y) trajectory in world metres; the subject of the path
        # viz. Final breakpoint pose is NOT recorded — see notes at EOF.
        self.path = []

        # Path-viz inputs handed in as plain data via set_map / set_astar_path.
        self._map = None
        self._start_xy = None
        self._goal_xy = None
        self._astar_path = None
        self._astar_path_length_m = None

    # --- Hooks subclasses override --------------------------------------------

    def _image_filename(self, cam_name, step):
        # Eval convention; DataGenRecorder overrides to GoalGraph's `NNN.png`.
        return f"step_{step:03d}.png"

    def _write_cam_images(self, step, observations):
        for cam_name, image in observations["images"].items():
            if cam_name in self.cam_dirs:
                Image.fromarray(image).save(self.cam_dirs[cam_name] / self._image_filename(cam_name, step))

    def _result_extras(self):
        # Empty by default; DataGenRecorder fills in data-gen metadata.
        return {}

    def _create_path_visualization(self, outcome):
        """Render the executed path (+ optional A* plan) over the env map to
        <run_dir>/rollout_viz.png, driven by run_cfg.recorder.path_viz.

        Needs set_map(...) to have been called; the A* overlay additionally
        needs set_astar_path(...). Rendering itself lives in planning_viz so
        this recorder only orchestrates (gathers data + picks options)."""
        from isaac_sim.utils.planning_viz import render_path_map

        if self._map is None:
            print("[RunRecorder] path_viz enabled but set_map() was never called "
                  "— skipping rollout viz.")
            return

        out = render_path_map(
            self._map,
            **self._path_map_kwargs(),
            title="A* plan vs robot rollout" + (f"   outcome={outcome}" if outcome else ""),
            out_path=self.run_dir / "rollout_viz.png",
        )
        print(f"[RunRecorder] rollout viz: {out}")

    def _path_map_kwargs(self):
        """render_path_map options: the cosmetics come from cfg.viz.path_map
        (single source of truth), the geometry from what we hold. Shared by the
        finalize rollout PNG and the per-step behaviour-panel map so the two stay
        visually identical. include_astar (a viz toggle) decides whether the A*
        layer is passed at all."""
        style = OmegaConf.select(self.run_cfg, "viz.path_map")
        # Draw the A* plan only if one was provided AND the config wants it.
        include_astar = bool(style["include_astar"]) if style is not None else True
        astar = self._astar_path if include_astar else None
        return {
            "style": style,
            "robot_path_xy": self.path,
            "astar_path_xy": astar,
            "start_xy": self._start_xy,
            "goal_xy": self._goal_xy,
        }

    # --- Public API -----------------------------------------------------------

    def set_map(self, occ_map, start_xy, goal_xy):
        """Hand the recorder the env's occupancy/SDF Map + start/goal poses
        (world metres) for the path viz. Plain data only (mirrors set_astar_path)."""
        self._map = occ_map
        self._start_xy = (float(start_xy[0]), float(start_xy[1]))
        self._goal_xy = (float(goal_xy[0]), float(goal_xy[1]))

    def set_astar_path(self, densified_path):
        """Store the A* plan (list of (x, y) world metres) for the path viz and,
        for DataGenRecorder, astar_path.json. Also computes its arc length."""
        self._astar_path = [(float(x), float(y)) for x, y in densified_path]
        self._astar_path_length_m = _path_length_m(self._astar_path)

    def record_step(self, step, observations, action, dist_to_goal, infer_ms=None):
        self._write_cam_images(step, observations)
        v, w = action
        entry = {
            "step": int(step),
            "x": float(observations["x"]),
            "y": float(observations["y"]),
            "heading_rad": float(observations["heading_rad"]),
            "v": float(v),
            "w": float(w),
            "dist_to_goal": float(dist_to_goal),
        }
        if infer_ms is not None:
            entry["infer_ms"] = round(float(infer_ms), 1)
        self.log_file.write(json.dumps(entry) + "\n")
        self.log_file.flush()
        self.n_steps = step + 1
        self.path.append((observations["x"], observations["y"]))

    def finalize(self, outcome, final_dist):
        self.log_file.close()
        result = {
            "outcome": outcome,
            "steps": int(self.n_steps),
            "final_dist": float(final_dist),
        }
        result.update(self._result_extras())
        (self.run_dir / "result.json").write_text(json.dumps(result, indent=2))
        print(f"[RunRecorder] {outcome}  steps={self.n_steps}  final_dist={final_dist:.2f}m")
        # Path viz is diagnostic-only: catch + print loudly rather than fail an
        # otherwise-good run. Default-False so configs omitting the block skip it.
        if bool(OmegaConf.select(self.run_cfg, "recorder.path_viz.enabled", default=False)):
            try:
                self._create_path_visualization(outcome)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[RunRecorder] rollout viz failed (non-fatal): {type(e).__name__}: {e}")
        return result
    
    def save_robot_behavior(self, cur_img, goal_img, goal_pose, waypoints,
                            linear_vel, angular_vel, metric_waypoint_spacing, mask_number, step):
        # Figure-level cosmetics from cfg.viz.behaviour_panel (single source).
        bp = self.run_cfg.viz.behaviour_panel
        panel_title_fs = float(bp["panel_title_fontsize"])
        fig = plt.figure(figsize=tuple(bp["figsize"]), dpi=float(bp["dpi"]))
        gs = fig.add_gridspec(2, 2)
        ax_ob = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_graph_pos = fig.add_subplot(gs[:, 1])

        ax_ob.imshow(np.array(cur_img).astype(np.uint8))

        # Bottom-left panel shows the goal image, UNLESS this step has no goal
        # image (the policy then feeds cur_img as a placeholder, so the panel
        # would just duplicate top-left) — in that case show map + path-so-far.
        # See notes at EOF for the mask → modality mapping.
        _GOAL_IMAGE_MASKS = {2, 3, 5, 6}  # masks that carry a goal image
        pv = OmegaConf.select(self.run_cfg, "recorder.path_viz", default=None)
        use_map_panel = (
            pv is not None
            and bool(pv.get("behavior_map_panel", True))
            and int(mask_number[0]) not in _GOAL_IMAGE_MASKS
            and self._map is not None
        )
        if use_map_panel:
            from isaac_sim.utils.planning_viz import render_path_map
            map_arr = render_path_map(
                self._map, **self._path_map_kwargs(),
                title="map + path so far", as_array=True,
            )
            ax_goal.imshow(map_arr)
            goal_panel_title = "Map + path so far (no goal image this step)"
        else:
            ax_goal.imshow(np.array(goal_img).astype(np.uint8))
            goal_panel_title = "Egocentric goal image"

        # Right panel: predicted action chunk in the robot-local frame. Shared
        # with the val-debug viz (inspect_val_disagreement); eval has no per-step
        # GT so gt_xy is omitted. Goal-pose star only for pose-carrying masks.
        from isaac_sim.utils.action_viz import plot_action_chunk
        mask_type = int(mask_number[0])
        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]
        modality_label = mask_texts[mask_type] if mask_type < len(mask_texts) else None
        _POSE_MASKS = {1, 3, 4, 5, 8}  # masks that carry a pose goal
        goal_for_plot = goal_pose if mask_type in _POSE_MASKS else None
        plot_action_chunk(
            ax_graph_pos,
            pred_xy=waypoints,                 # reads cols 0 (x=fwd), 1 (y=left)
            style=self.run_cfg.viz.action_chunk,
            goal_pose_loc_norm=goal_for_plot,
            modality_label=modality_label,
            title="Normalized generated 2D trajectories from OmniVLA",
        )

        ax_ob.set_title("Egocentric current image", fontsize=panel_title_fs)
        ax_goal.set_title(goal_panel_title, fontsize=panel_title_fs)

        # Lazy-create behaviour/ on first call so unused runs leave no empty dir.
        behaviour_dir = self.run_dir / "behaviour"
        behaviour_dir.mkdir(parents=True, exist_ok=True)
        save_path = behaviour_dir / f"step_{step:03d}.jpg"
        plt.savefig(save_path)
        plt.close(fig)


def _git_sha():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _path_length_m(path_xy):
    # Sum of consecutive segment lengths. path_xy is a list of (x, y) tuples.
    # e.g. [(0,0), (1,0), (1,1)] → 1.0 + 1.0 = 2.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(path_xy[:-1], path_xy[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


class DataGenRecorder(RunRecorder):
    """Data-gen recorder: writes a GoalGraph-compatible frames.json + image
    directory, persists the A* path, and adds reproducibility metadata.

    Output layout is documented in the notes block at the end of this file.
    """

    def __init__(self, run_cfg, run_name):
        super().__init__(run_cfg, run_name)

        # Which camera becomes the GoalGraph node image. Owned by the TASK
        # config (a property of the task, not the recorder). Fallback covers
        # data_gen runs where task != image_graph; see notes at EOF.
        self.goal_graph_cam_name = getattr(
            run_cfg.task, "goal_graph_camera_name", "front_cam"
        )

        # Rename the goal-graph cam's dir to GoalGraph's canonical `frames/` so
        # GoalGraph.image_dir_for(frames.json) resolves without a path rewrite.
        if self.goal_graph_cam_name in self.cam_dirs:
            new_dir = self.run_dir / "frames"
            self.cam_dirs[self.goal_graph_cam_name].rename(new_dir)
            self.cam_dirs[self.goal_graph_cam_name] = new_dir
        else:
            # Cam not in logging set — still need frames/ for the node images.
            # Add it to cam_dirs so _write_cam_images picks it up.
            print(f"[DataGenRecorder] WARN: goal_graph_cam_name={self.goal_graph_cam_name!r} "
                  f"is not in env.cameras with logging=True; forcing it into the logged set "
                  f"because the GoalGraph needs its frames.")
            frames_dir = self.run_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            self.cam_dirs[self.goal_graph_cam_name] = frames_dir

        self._graph_nodes = []  # GoalGraph node accumulator

        # Snapshot fields needed at finalize so finalize() doesn't depend on
        # run_cfg sticking around.
        self._env_config_name = run_cfg.task.name
        self._env_config_hash = hashlib.sha1(
            OmegaConf.to_yaml(run_cfg.env).encode("utf-8")
        ).hexdigest()[:10]
        self._robot_inflation_m = float(run_cfg.env.astar.robot_inflation_m)
        # goal_radius_m / standoff_m left at GoalGraph's default — no consumer.
        self._target_speed_mps = float(run_cfg.controller.max_linear_vel_mps)
        # Sample cadence is sim-time: env.step() advances 1/record_hz of sim
        # time per call. See notes at EOF.
        self._sample_hz = float(run_cfg.record_hz)

    # --- Hook overrides -------------------------------------------------------

    def _image_filename(self, cam_name, step):
        # GoalGraph convention: 3-digit zero-padded node index. e.g. 0 → "000.png"
        return f"{step:03d}.png"

    def _result_extras(self):
        return {
            "n_frames": len(self._graph_nodes),
            "git_sha": _git_sha(),
            "astar_path_length_m": self._astar_path_length_m,
            "sample_hz": self._sample_hz,
            "target_speed_mps": self._target_speed_mps,
        }

    # --- Public API extensions -----------------------------------------------

    def record_step(self, step, observations, action, dist_to_goal, infer_ms=None):
        # Parent writes the image (our naming) + jsonl; we also accumulate a
        # GraphNode for the GoalGraph emission at finalize().
        super().record_step(step, observations, action, dist_to_goal, infer_ms)
        from isaac_sim.planning.a_star.goal_graph import GraphNode
        v, w = action
        # image_filename matches what _image_filename(...) wrote — same `step`.
        self._graph_nodes.append(GraphNode(
            xy=(float(observations["x"]), float(observations["y"])),
            yaw_rad=float(observations["heading_rad"]),
            image_filename=self._image_filename(self.goal_graph_cam_name, step),
            cmd_v=float(v),
            cmd_w=float(w),
        ))

    def finalize(self, outcome, final_dist):
        # 1. Persist the A* path (or warn loudly if data_gen forgot to set it).
        if self._astar_path is not None:
            (self.run_dir / "astar_path.json").write_text(json.dumps({
                "path_xy": [list(p) for p in self._astar_path],
                "path_length_m": self._astar_path_length_m,
                "n_waypoints": len(self._astar_path),
            }, indent=2))
        else:
            print("[DataGenRecorder] WARN: set_astar_path() was never called — "
                  "astar_path.json will be missing from this run.")

        # 2. Emit the GoalGraph manifest before super().finalize().
        from isaac_sim.planning.a_star.goal_graph import GoalGraph
        graph = GoalGraph(
            env_config_name=self._env_config_name,
            env_config_hash=self._env_config_hash,
            node_spacing_m=0.0,             # rollout mode: spacing varies — sentinel "n/a"
            robot_inflation_m=self._robot_inflation_m,
            sampling_mode="policy_rollout",  # distinct from old "realtime"
            target_speed_mps=self._target_speed_mps,
            sample_hz=self._sample_hz,
            nodes=self._graph_nodes,
        )
        graph.save(self.run_dir / "frames.json")

        # 3. Parent writes result.json (with our extras via _result_extras).
        return super().finalize(outcome, final_dist)
