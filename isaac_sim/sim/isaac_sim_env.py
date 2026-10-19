import os
import sys

# Path setup so sibling + scenes imports resolve.
# File lives at isaac_sim/sim/isaac_sim_env.py; two parent hops -> repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, f"{_REPO_ROOT}/isaac_sim", os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# NOTE: anything importing omni / pxr / isaacsim must be imported AFTER
# SimulationApp boots — those are deferred to inside __init__. See notes at EOF.
from isaac_sim.planning.a_star.planner import A_star
from isaacsim import SimulationApp
from scenes import _common as env_common
from scenes.templates import REGISTRY as TEMPLATE_REGISTRY
from omegaconf import OmegaConf
import math

# Process-level singletons — see notes at EOF for why these are module-scope.
_SIM_APP = None   # SimulationApp; boot once, reuse across all envs.
_LAST_ENV = None  # most-recent IsaacSimEnv, for ordered teardown on RELOAD.


def _ensure_sim_app(headless=True):
    """Starts the simulation app and rewrites cache since we have Sudo issues."""
    global _SIM_APP
    if _SIM_APP is not None:
        return _SIM_APP
    # Redirect Kit caches to user-writable dirs (/isaacsim is root-owned).
    # OV_CACHE_DIR gives parallel data-gen instances separate caches; see EOF.
    _ov_cache = os.path.expanduser(os.environ.get("OV_CACHE_DIR", "~/.cache/ov-isaacsim"))
    _ov_logs = os.path.join(_ov_cache, "logs")
    _ov_shaderdb = os.path.join(_ov_cache, "shaderdb")
    for _d in (_ov_cache, _ov_logs, _ov_shaderdb,
               os.path.join(_ov_cache, "shaders/nv_shadercache"),
               os.path.join(_ov_cache, "shadercache/common"),
               os.path.join(_ov_cache, "DerivedDataCache")):
        os.makedirs(_d, exist_ok=True)
    _extra_args = [
        f"--/app/tokens/omni_cache={_ov_cache}",
        f"--/app/tokens/cache={_ov_cache}",
        f"--/app/tokens/logs={_ov_logs}",
        f"--/rtx/shaderDb/shaderCachePath={_ov_shaderdb}",
    ]
    # OV_THREAD_COUNT caps Kit's task threads so N parallel instances don't
    # oversubscribe the CPU during shader compiles. Unset = Isaac default (32).
    _n_threads = os.environ.get("OV_THREAD_COUNT")
    if _n_threads:
        _extra_args += [
            f"--/plugins/carb.tasking.plugin/threadCount={_n_threads}",
            f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={_n_threads}",
        ]
    _SIM_APP = SimulationApp({"headless": headless, "extra_args": _extra_args})
    return _SIM_APP


class IsaacSimEnv:
    """One IsaacSim environment isntance which can be used for collecting rollouts in evaluation or data generaiton."""

    def __init__(self, env_cfg, headless=True, record_hz=3.0, physics_hz=60):
        """
        initializes on IsaacSim env with the given arguments for the env

        Args:
            env_cfg: a config file containing all important arguments on the scene construction, robot, and cameras.
            headless: whether to boot in headless or not.
            record_hz: the rate at which we record data from the environment (required for recorder) not part of env_cfg since recorder also uses it and we want single source of truth
            physics_hz: inner PhysX integrator rate (Hz). Sets World(physics_dt=1/physics_hz) and STEPS_PER_TICK. Passed in (not from env_cfg) for the same single-source reason as record_hz — it's a top-level cfg knob shared by eval + data_gen. NOT a tuning knob (60 Hz Isaac default; see eval.yaml).
        """
        # First env → boot Kit + fresh stage. Subsequent ones (RELOAD) reuse
        # SimulationApp but must wipe the previous stage + World singleton.
        global _LAST_ENV
        is_reload = _SIM_APP is not None
        self.sim_app = _ensure_sim_app(headless)
        # Deferred imports — must come AFTER SimulationApp boot.
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.nucleus import get_assets_root_path
        from scenes import _common as env_common
        from isaac_sim.sim.robot import Robot
        from isaac_sim.planning.a_star.map import MapBuilder
        from isaac_sim.sim.camera import Camera

        if is_reload:
            # Teardown order is load-bearing — see notes at EOF. Release the OLD
            # env's Kit resources, drop the World singleton, then wipe the stage.
            if _LAST_ENV is not None:
                _LAST_ENV._release_kit_resources()
            World.clear_instance()
            omni.usd.get_context().new_stage()

        self.env_cfg = env_cfg
        self.headless = headless

        # --- Base-USD scenes (the 2x3 warehouse) --------------------------------
        # If the scene config names a `base_usda`, open it AS the stage instead of
        # building the warehouse procedurally — loads a pre-authored layout
        # (reshape + manual tile fixes + CCTV cameras) verbatim from the file.
        # GATED: `base_usda` is unset for every existing scene, so their behaviour
        # is byte-identical. World() then wraps whatever stage is current.
        try:
            base_usda = env_cfg.get("base_usda", None)
        except AttributeError:
            base_usda = getattr(env_cfg, "base_usda", None)
        self._base_usda = base_usda
        if base_usda:
            import pathlib
            ok = omni.usd.get_context().open_stage(str(pathlib.Path(base_usda)))
            print(f"[IsaacSimEnv] base scene {base_usda} opened (ok={ok})")

        physics_dt = 1.0 / physics_hz  # inner PhysX dt; physics_hz is the single-source cfg knob (see eval.yaml / data_gen.yaml)
        self.world = World(stage_units_in_meters=1.0, physics_dt=physics_dt)
        self.assets_root = get_assets_root_path()
        self.STEPS_PER_TICK = round(1.0 / (record_hz * physics_dt))  # derived from cfg.record_hz + physics_hz

        if self.assets_root is None:
            raise RuntimeError("assets_root is none")

        # Convert Hydra DictConfig → template dataclass (runs __post_init__,
        # exposes computed properties like goal_xy that env_common.build expects).
        raw = OmegaConf.to_container(env_cfg, resolve=True)
        raw.pop("robot", None)
        raw.pop("cameras", None)
        raw.pop("astar", None)  # A* knobs live at cfg.env.astar, not on the template dataclass
        template_cfg = TEMPLATE_REGISTRY[raw["template"]].Config.from_dict(raw)
        env_common.build(self.world, self.assets_root, template_cfg)
        self.template_cfg = template_cfg

        # Build the occupancy map and A* planner.
        self.map_builde = MapBuilder(env_cfg.astar, self)
        self.map = self.map_builde.construct_map()
        self.a_star = A_star(env_cfg.astar, self.map)
        # Pass-1 planner: ALWAYS the target-less map. resolve_effective_* plans
        # to the goal CENTRE, which sits inside a /World/Goal target prop — so
        # the resolver must keep using this planner even after the swap below.
        self._resolve_planner = self.a_star

        # Resolve start + goal poses from a single A* plan. Must run AFTER
        # A_star (resolver calls into it) and BEFORE Robot (spawn pose feeds it).
        self.effective_start_pose = None
        self.effective_goal_pose = None
        self.resolve_effective_start_and_goal_pose()

        # TWO-PASS PLANNING (2026-07-20, Lukas's standoff design + graze fix):
        # pass 1 above ignored the /World/Goal target (needed: its centre IS the
        # goal), which left the driven path with no clearance guarantee from the
        # target itself — measured 0.37 m surface / ~7 cm chassis margin on a
        # forklift target (potential_bugs 2026-07-20). Fix: after the effective
        # (standoff) goal is fixed, swap the PUBLIC planner to a target-aware
        # map, so the start->effective-goal plan (data_gen.py pass 2) avoids the
        # target like any other obstacle. Requires standoff_buffer_m >
        # robot_inflation_m, else the effective goal sits inside the inflated
        # target and pass 2 cannot plan (warehouse_2x3.yaml: 0.5 > 0.35).
        goal_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Goal")
        if goal_prim.IsValid():
            from pxr import Usd, UsdGeom
            from isaac_sim.planning.a_star.map import Map, euclidean_sdf
            bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   includedPurposes=[UsdGeom.Tokens.default_])
            target_map = Map(self.map.occupancy.copy(), None,
                             self.map.origin, self.map.resolution)
            target_map.add_prim(goal_prim, bc,
                                float(env_cfg.astar.max_obstacle_half_extent_m))
            target_map.sdf = euclidean_sdf(target_map.occupancy, target_map.resolution)
            self.map = target_map
            self.a_star = A_star(env_cfg.astar, target_map)
            print("[IsaacSimEnv] two-pass planning: /World/Goal target added to "
                  "the rollout map (pass-2 plans avoid it; standoff endpoint from pass 1)")

        # Spawn the robot at the effective start pose. start_xy is authoritative;
        # only the spawn YAW is derived from the A* path. wxyz quat about world +Z.
        import numpy as np
        _spx, _spy = self.effective_start_pose["xy"]
        _spyaw = self.effective_start_pose["heading_rad"]
        _initial_pos = np.array([_spx, _spy, 0.0], dtype=np.float64)
        _initial_ori = np.array(
            [math.cos(_spyaw / 2), 0.0, 0.0, math.sin(_spyaw / 2)],
            dtype=np.float64,
        )
        self.robot = Robot(env_cfg.robot, (_initial_pos, _initial_ori), self.assets_root)

        self.world.scene.add(self.robot.object)  # Robot exposes the WheeledRobot as .object

        stage = omni.usd.get_context().get_stage()
        self.cameras = {}
        for cam_name, cam_cfg in env_cfg.cameras.items():
            self.cameras[cam_name] = Camera(cam_cfg, stage, world=self.world)

        self.task = None  # set via set_task before record_image_graph

        # Register as the latest live env so the next RELOAD can release our
        # Kit-side resources before wiping the stage.
        _LAST_ENV = self  # noqa: F841 — visible to module via `global` above

    def _release_kit_resources(self):
        """Releases the resources occupied by the enviroment, i.e. destroys the cameras and removes the robot."""
        for cam_name, cam in list(self.cameras.items()):
            try:
                cam.destroy()
            except Exception as e:
                print(f"[IsaacSimEnv] _release_kit_resources: "
                      f"camera {cam_name!r} destroy raised {e!r} — ignoring")
        self.cameras.clear()
        # Drop the robot ref so late finalizers don't trip on the WheeledRobot's
        # articulation handle after the stage is wiped.
        self.robot = None

    def get_observations(self, capture_images=True):
        """Returns the current observations from the environment.

        Args:
            capture_images: when False, skip the per-camera GPU->host readback
                (cam.capture()) and return images={}; pose is still populated.
                Used on non-recording control ticks during data-gen, where the
                oracle reads pose only and the frames are never recorded — so the
                rendered scene is unchanged and recorded frames stay byte-identical.
        Returns:
            dict: dictionary with images, which is a dict for each camera one observation, the x and y coordinates and the heading radian of the robot.
        """
        pos, yaw = self.robot.get_world_pose(ori_to_yaw=True)
        images = {cam_name : cam.capture() for cam_name, cam in self.cameras.items()} if capture_images else {}
        return {
            "images" : images,
            "x" : float(pos[0]),
            "y" : float(pos[1]),
            "heading_rad": float(yaw)
        }

    def reset(self):
        """Resets the environment by resetting the world, the robot and renders a few ticks.
        
        Returns:
            dict: the initial observations after reset of the env.
        """
        self.world.reset()
        self.robot.reset()
        # Warm-up ticks — without them there's no image input on the first step.
        for _ in range(self.STEPS_PER_TICK):
            self.world.step(render=True)
        return self.get_observations()

    def get_effective_goal_pose(self):
        """Return the effective goal pose which is calculated in resolve effective_start_and_goal_pose."""
        # Eagerly populated by resolve_effective_start_and_goal_pose() in __init__.
        return self.effective_goal_pose

    def get_a_start_path(self, start, goal):
        """Returns the A* path used to decide effective goal pose."""
        return self.a_star.get_a_star_path(start, goal)


    def resolve_effective_start_and_goal_pose(self):
        """Resolves effective start and goal pose from calculating an A* path and then orienting the robot so that it is aligned with the path.
        
        This function has no input, after calling it it will calculate an A* path and then set 
        the start so that it is aligned with the initial trajecotry and the end is aligned 
        with the last few steps taken so no turning is required at the first or last postion."""
        import omni.usd  # deferred — Kit must be booted

        sx, sy = self.template_cfg.start_xy
        gx, gy = self.template_cfg.goal_xy
        ac = self.env_cfg.astar

        # A* sampled at grid_resolution_m → uniform-arclength list of (x, y).
        # We average K interior secants for the endpoint headings (rather than
        # asking the Path for its endpoint tangent) because natural-spline
        # boundaries "straighten" the tangent near the endpoints; secant
        # averaging is more robust against that artefact.
        # _resolve_planner, NOT self.a_star: pass 1 must ignore a /World/Goal
        # target (goal centre sits inside it) — see the two-pass block in __init__.
        path = self._resolve_planner.get_a_star_path((sx, sy), (gx, gy)).sample(
            float(ac.grid_resolution_m)
        )
        K = min(int(ac.heading_window_segments), len(path) - 1)

        # Spawn heading: first K segments. Symmetric with the goal end below.
        p_start = path[0]
        p_forward = path[K]
        astar_heading = math.atan2(p_forward[1] - p_start[1],
                                   p_forward[0] - p_start[0])

        # Mode dispatch (start_heading_mode: astar | fixed | random). Default
        # "astar" reproduces legacy behavior exactly; "fixed"/"random" stop
        # the A* answer from leaking into the spawn yaw (eval probes).
        mode = getattr(self.template_cfg, "start_heading_mode", "astar")
        start_heading = env_common.resolve_start_heading_rad(
            mode,
            self.template_cfg.start_heading_deg,
            self.template_cfg.seed,
            astar_heading,
        )
        if mode != "astar":
            print(f"[IsaacSimEnv] start_heading_mode={mode}: spawn yaw "
                  f"{math.degrees(start_heading):.1f} deg "
                  f"(astar would be {math.degrees(astar_heading):.1f} deg)")

        # Approach heading: last K segments. 8-conn A* yields 45°-multiples
        # per raw segment; averaging over K samples smooths that out.
        p_end = path[-1]
        p_back = path[-K - 1]
        goal_heading = math.atan2(p_end[1] - p_back[1],
                                  p_end[0] - p_back[0])

        # No-prop scenes (template1 with goal_kinds=[]) skip the standoff —
        # the goal is a point in space, robot drives straight to it.
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/Goal").IsValid():
            prop_radius = self._read_goal_prop_radius(stage)
            standoff = max(prop_radius + float(ac.standoff_buffer_m),
                           float(ac.min_standoff_m))
        else:
            standoff = 0.0

        # Walk `standoff` m back from the centroid along the approach direction.
        eff_gx = gx - standoff * math.cos(goal_heading)
        eff_gy = gy - standoff * math.sin(goal_heading)

        # Goal-pose heading FED TO THE POLICY. The A* approach tangent
        # (goal_heading) encodes the detour side — an L-wall forces the route
        # around the OPEN end, so its sign is the which-way-around answer.
        # Handing that to the pose input leaks the solution (the ego-only
        # baseline solved L purely off this sign). Wherever the spawn yaw is
        # de-leaked (start_heading_mode != "astar" — the eval probes), de-leak
        # the goal yaw the SAME way: point it "straight", i.e. at the start
        # heading, so the goal pose carries NO side information and the policy
        # must see the route. The physical standoff (eff_gx/eff_gy above) still
        # uses the true approach tangent; legacy "astar" scenes are unchanged.
        goal_pose_heading = goal_heading if mode == "astar" else start_heading
        self.effective_start_pose = {"xy": (sx, sy),
                                     "heading_rad": start_heading}
        self.effective_goal_pose = {"xy": (eff_gx, eff_gy),
                                    "heading_rad": goal_pose_heading}
        return {"start": self.effective_start_pose,
                "goal": self.effective_goal_pose}

    def _read_goal_prop_radius(self, stage):
        """Reads the goal prop radius from the stage, necessary for standoff calculations in case we use goal propos."""
        from pxr import Usd, UsdGeom

        goal_prim = stage.GetPrimAtPath("/World/Goal")
        if not goal_prim.IsValid():
            raise RuntimeError("/World/Goal not found in stage")
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_],
        )
        aabb = bbox_cache.ComputeWorldBound(goal_prim).ComputeAlignedRange()
        mn, mx = aabb.GetMin(), aabb.GetMax()
        rx = 0.5 * (mx[0] - mn[0])
        ry = 0.5 * (mx[1] - mn[1])
        return float(math.hypot(rx, ry))

    def step(self, actions, n_physics_ticks=None, capture_images=True):
        """Applies one step, which is according to the physics steps. Recorder and Controller can have different frequency for updates.

        capture_images is forwarded to get_observations(): data-gen passes False
        on non-recording ticks to skip the camera readback.

        RENDER: we MUST call world.step(render=True) on every physics tick.

        !! DO NOT "OPTIMISE" THIS BY RENDERING ONLY BEFORE THE READBACK !!
        Tried and REVERTED 2026-07-13. The RTX render is the dominant GPU cost of
        a sim-in-the-loop eval (4x 720x720 CCTV + 640x480 ego ~= 2.4M px/frame)
        and images are only read back at 3 Hz while this loop runs at 30 Hz, so
        gating the render on `capture_images` looks like a free ~20x saving.
        It is not: Isaac's camera annotators need MORE THAN ONE render pass to
        refresh, so a single render before the readback hands back the PREVIOUS
        tick's frame. Measured (tests/test_render_gating.py): the gated frame at
        step N was 6x closer to the legacy frame at step N-1 than to its own
        step N (mean|diff| 1.8 vs 11.1) => the policy silently observes a world
        one control step (~333 ms) STALE. It does not crash and does not look
        wrong; it would just quietly corrupt every eval and every generated
        dataset.

        The sanctioned retry is ISAAC_RENDER_FLUSH_K=K: skip the render on
        every physics tick and instead issue K physics-free World.render()
        passes (playSimulations off -> app.update -> on) immediately before
        the readback, flushing the annotator pipeline at the CURRENT state.
        (A "render the last K ticks" variant does NOT work: the 30 Hz loops
        call env.step with n_physics_ticks=2, so per-call gating saturates at
        2 renders — and K=2 measured STALE, the pipeline is >=2 frames deep.)
        Any K MUST pass tests/test_render_gating.py, which compares each frame
        against the reference's CURRENT vs PREVIOUS frame (a plain pixel-diff
        cannot see staleness: the sim is nondeterministic, so even
        legacy-vs-legacy is 0/100 pixel-identical). Default (unset/0) = legacy.

        MEASURED 2026-07-20 (outputs/render_gate_verdict*.txt): K=3 and K=4
        -> STALE (early frames match reference N-1); K=8 -> 5 fresh / 14
        ambiguous, while the legacy-vs-legacy CONTROL is 19/19 fresh with 0
        ambiguous — so K=8 is NOT observation-equivalent either (partially
        refreshed annotator state, not clean staleness). K=16 would save only
        ~20% of renders. VERDICT: flush-K FAILED validation at every useful K;
        leave ISAAC_RENDER_FLUSH_K unset. Wall-clock speed for big campaigns
        comes from parallel data-gen lanes (4x validated 2026-07-20, 5.2 GB
        VRAM/instance), not render gating.
        """
        # Default (n_physics_ticks=None) → one recording tick (STEPS_PER_TICK
        # physics ticks = one 3 Hz VLA-inference interval). Eval uses the default.
        # data_gen.py overrides with n_physics_ticks=steps_per_control so each
        # env.step() advances one fast control tick; total physics work per
        # recorded sample is identical (decimation × steps_per_control == STEPS_PER_TICK).
        if n_physics_ticks is None:
            n_physics_ticks = self.STEPS_PER_TICK
        self.robot.apply_action(actions)
        # ISAAC_RENDER_FLUSH_K (see banner above): 0/unset = legacy every-tick
        # render; K>=1 = physics ticks run UNrendered, then K physics-free
        # World.render() passes flush the annotator pipeline at the current
        # state right before the readback (no renders at all on
        # capture_images=False calls). Any K must pass
        # tests/test_render_gating.py before real use.
        flush_k = int(os.environ.get("ISAAC_RENDER_FLUSH_K", "0"))
        for _ in range(n_physics_ticks):
            self.world.step(render=(flush_k <= 0))
        if flush_k > 0 and capture_images:
            for _ in range(flush_k):
                self.world.render()
        return self.get_observations(capture_images=capture_images)

    def set_start_pose(self, start_xy, start_heading_deg=None,
                       start_heading_mode=None):
        """Re-pose the robot's START without rebuilding the env ("fast grid").

        For pose sweeps over a FIXED scene (wall_layout grids), recreating
        the whole env per run costs ~8 min while only the spawn changes.
        This updates the three places the spawn pose lives, so the next
        env.reset() (every eval run begins with one) places the robot at the
        new pose via the existing Robot.reset() path:

          1. template_cfg.start_* — read by the recorder map viz and the
             resolver below. NOTE: set EXACT values; the template's seeded
             jitter ran at config time and does NOT re-apply here.
          2. resolve_effective_start_and_goal_pose() — re-plans A* from the
             new start (spawn yaw per start_heading_mode, approach heading /
             effective goal pose depend on the path end).
          3. robot.initial_pos / initial_ori — what Robot.reset() teleports
             to (world.reset alone would restore the construction pose).

        Only meaningful for explicit-spawn templates (wall_layout, t1/t2,
        random_open). Driven by eval_policy.py for KEEP_ENV/KEEP_BOTH runs
        that carry pose-override keys.
        """
        import numpy as np

        self.template_cfg.start_xy = (float(start_xy[0]), float(start_xy[1]))
        if start_heading_deg is not None:
            self.template_cfg.start_heading_deg = float(start_heading_deg)
        if start_heading_mode is not None:
            self.template_cfg.start_heading_mode = str(start_heading_mode)

        self.resolve_effective_start_and_goal_pose()

        spx, spy = self.effective_start_pose["xy"]
        spyaw = self.effective_start_pose["heading_rad"]
        self.robot.initial_pos = np.array([spx, spy, 0.0], dtype=np.float64)
        self.robot.initial_ori = np.array(
            [math.cos(spyaw / 2), 0.0, 0.0, math.sin(spyaw / 2)],
            dtype=np.float64,
        )
        print(f"[IsaacSimEnv] set_start_pose: start=({spx:.2f},{spy:.2f}) "
              f"yaw={math.degrees(spyaw):.1f} deg (env NOT rebuilt)")

    def teleport(self, xy, yaw_rad):
        """Teleports the robot to a new position. Used for recoridng the image goal graph."""
        import numpy as np
        position = np.array([xy[0], xy[1], 0.0], dtype=np.float64)
        # wxyz quat about world +Z — matches the spawn-pose convention in __init__.
        orientation = np.array(
            [math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)],
            dtype=np.float64,
        )
        self.robot.object.set_world_pose(position=position, orientation=orientation)
        # set_world_pose doesn't reset velocity — without these, pre-teleport
        # momentum leaks into the next physics step.
        self.robot.object.set_linear_velocity(np.zeros(3))
        self.robot.object.set_angular_velocity(np.zeros(3))
        for _ in range(3):
            self.world.step(render=True)

    def set_task(self, task):
        """Sets the taks for the environment, used to decide whether we have to record an image goal graph."""
        self.task = task
        if self.task.goal_graph_path:  # truthy = image-graph task
            self._record_image_graph()

    def _record_image_graph(self):
        """Records the image goal grpah for the robot's task.
        
        This function teleports the robot along the A* path, caputes an image with the camera that is used for navigation and mvoes on to the next postion."""
        from pathlib import Path
        from PIL import Image
        from isaac_sim.planning.a_star.goal_graph import (
            GoalGraph, GraphNode,
        )

        cam = self.cameras[self.task.goal_graph_camera_name]
        # CCTV goal frames: also capture cctv view at each node so use_cctv_goal
        # policies (run3/4/5) find <graph_dir>/cctv/NNN.png. Harmless if absent.
        cctv_cam = self.cameras["cctv_topdown"] if "cctv_topdown" in self.cameras else None
        cctv_dir = None
        # Runs from set_task, BEFORE env.reset() is called externally. World
        # needs reset before step() (physics init), and the camera annotator
        # needs ~STEPS_PER_TICK render ticks before the first frame is valid
        # (else cam.capture() raises "No frame captured"). Mirror reset()'s init.
        self.world.reset()
        self.robot.reset()
        for _ in range(self.STEPS_PER_TICK):
            self.world.step(render=True)
        # Plan straight to the EFFECTIVE goal, sampled at node_spacing_m (paper's
        # training cadence ~1 Hz × 0.5 m/s = 0.5 m). Heading convention is
        # delegated to the Path subclass: SplinePath gives the analytic tangent,
        # PolylinePath the secant between consecutive samples.
        planned_path = self.get_a_start_path(
            self.template_cfg.start_xy,
            self.get_effective_goal_pose()["xy"],
        )
        nodes_xy, node_yaws = planned_path.sample_with_heading(
            float(self.task.goal_graph_node_spacing_m)
        )

        output_json = Path(self.task.goal_graph_path)
        image_dir = output_json.with_suffix("")  # sibling dir, name minus .json
        image_dir.mkdir(parents=True, exist_ok=True)
        for stale in sorted(image_dir.glob("*.png")):
            stale.unlink()
        if cctv_cam is not None:
            cctv_dir = image_dir / "cctv"
            cctv_dir.mkdir(parents=True, exist_ok=True)
            for stale in sorted(cctv_dir.glob("*.png")):
                stale.unlink()

        graph = GoalGraph(
            env_config_name=getattr(self.template_cfg, "name", "unknown"),
            env_config_hash="",
            node_spacing_m=float(self.task.goal_graph_node_spacing_m),
            robot_inflation_m=float(self.env_cfg.astar.robot_inflation_m),
            sampling_mode=self.task.goal_graph_sampling_mode,
            nodes=[],
        )

        for i, (xy, yaw) in enumerate(zip(nodes_xy, node_yaws)):
            self.teleport(xy, yaw)
            rgb = cam.capture()
            fname = f"{i:03d}.png"
            Image.fromarray(rgb).save(image_dir / fname)
            if cctv_cam is not None:
                Image.fromarray(cctv_cam.capture()).save(cctv_dir / fname)
            graph.nodes.append(GraphNode(
                xy=(float(xy[0]), float(xy[1])),
                yaw_rad=float(yaw),
                image_filename=fname,
            ))

        graph.save(output_json)
        print(f"[env] recorded image graph: {len(nodes_xy)} nodes → {output_json}")
        return graph
