from isaac_sim.utils.controller_utils import solve_twist, world_to_robot


class WaypointFollower:
    """Steers toward ONE world-frame target waypoint (the lookahead carrot,
    chunk index waypoint_select), recomputing (v, w) from the live pose every
    control tick. The carrot is refreshed on each ~3 Hz model query."""

    def __init__(self, controller, model_query_hz):
        """
        Args:
            controller: cfg.controller block (max_linear_vel_mps,
                max_angular_vel_radps, waypoint_select, eps).
            model_query_hz: rate the VLA is re-queried (= eval control_hz, 3 Hz).
        """
        self.controller = controller
        self.horizon_s = float(controller.waypoint_select + 1) / float(model_query_hz)
        self.target = None  # world-frame (x, y) carrot, set per model query

    def set_target(self, target_xy):
        """Install the carrot (called on each ~3 Hz model query)."""
        self.target = target_xy

    def __call__(self, obs):
        """Live pose -> (v, w) toward the fixed carrot. No arrival brake: the
        model decelerates near the goal by predicting a nearer waypoint (dx -> 0
        -> v -> 0), and the eval loop's success-dist check terminates the run."""
        dx, dy = world_to_robot(self.target[0], self.target[1],
                                obs["x"], obs["y"], obs["heading_rad"])
        return solve_twist(
            dx, dy,
            self.horizon_s,
            float(self.controller.eps),
            float(self.controller.max_linear_vel_mps),
            float(self.controller.max_angular_vel_radps),
        )
