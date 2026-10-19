import math

from isaac_sim.utils.controller_utils import (
    pure_pursuit_target,
    solve_twist,
    world_to_robot,
)


class A_star_policy:
    """A policy that uses a pre-computed A* path to generate the next action."""
    VLA_MODEL_SAMPLE_HZ: float = 3.0

    def __init__(self, cfg, controller, control_hz):
        """Initializes a new policy instance."""
        self.cfg = cfg
        self.controller = controller
        self.dt_s = 1.0 / float(control_hz)

        self.lookahead_m = (
            float(self.controller.max_linear_vel_mps)
            * float(self.controller.waypoint_select + 1)
            / float(self.VLA_MODEL_SAMPLE_HZ)
        )
        self.arrival_distance_m = self.lookahead_m * float(cfg.arrival_distance_ratio)

        # Time horizon for the (v, w) command = time to traverse to the
        # lookahead at cruise speed. The command formula below is "reach the
        # lookahead in `horizon_s` seconds". Using 1/control_hz here (the old
        # behaviour) meant "reach it in ONE control tick" — at the 30 Hz
        # data-gen loop that demanded ~50 m/s and pinned w to its cap, so the
        # tracker limit-cycled (visible zigzag). horizon_s is independent of
        # control_hz (fix, 2026-06-15).
        self.horizon_s = self.lookahead_m / float(self.controller.max_linear_vel_mps)

        self.path = None
        self.curr = 0

    def set_path(self, path):
        """Sets the A* path for the policy to follow."""
        self.path = path
        self.curr = 0

    def __call__(self, obs, step):
        """"
        Call the policy with observations and current step

        Args:
            obs: dictionary of observations including the pose of the robot and images
            step: current step of the training process
        
        Returns:
            linear_vel_value_limit: linear volicty limited by caps
            angular_vel_value_limi: angular velocity also limited by cap
        """
        x = obs["x"]
        y = obs["y"]
        yaw = obs["heading_rad"]

        target_xy, _cursor = self._find_goal_node((x, y))
        tx, ty = target_xy

        # World->robot frame, then the shared pure-pursuit twist law (forward-only
        # + arctan2 + joint saturation). Identical math to the VLA follower; see
        # controller_utils.solve_twist. The arrival brake (oracle-specific) is
        # applied by shrinking the saturation box BEFORE the solve.
        dx, dy = world_to_robot(tx, ty, x, y, yaw)

        maxv, maxw = self.controller.max_linear_vel_mps, self.controller.max_angular_vel_radps

        gx, gy = self.path[-1]
        dist_to_goal = math.hypot(gx - x, gy - y)
        if dist_to_goal < self.arrival_distance_m:
            brake = max(0.0, dist_to_goal / self.arrival_distance_m)
            maxv *= brake
            maxw *= brake

        v, w = solve_twist(dx, dy, self.horizon_s, self.controller.eps, maxv, maxw)
        return v, w, {}

    def _find_goal_node(self, pos):
        """
        finds the next node to follow for pure prusit controller.

        Looks along the path for the next node that is above the lookahead distance but as little as possible abvoe that.
        Thin wrapper over the shared pure_pursuit_target (controller_utils) — kept
        so the existing per-method tests still exercise this entry point, and so
        the monotonic cursor lives on the policy instance.
        """
        target, cursor = pure_pursuit_target(self.path, pos, self.curr, self.lookahead_m)
        self.curr = cursor
        return target, cursor
