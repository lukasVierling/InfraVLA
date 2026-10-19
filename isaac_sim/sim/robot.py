import numpy as np

# isaacsim imports require SimulationApp to already be booted in this process.
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController


_SUPPORTED_CONTROLLERS = {"Differential"}


class Robot:
    """Spawns a robot in the environment."""

    def __init__(self, cfg, initial, assets_root):
        """Intialzes the robot.
        
        Args:
            cfg: robot config including name and wheel parameters etc.
            initial: the initila poisiton of the robot in the world
            assets_root: root path to the USD assets to get the robot file.
        """
        self.cfg = cfg
        self.controller = None
        self.object = None
        self.camera = None
        self.assets_root = assets_root
        self.initial_pos, self.initial_ori = initial

        # Single source of truth for the robot's stage path (read by camera
        # auto-discovery, map obstacle exclusion, ...).
        self.prim_path = cfg.prim_path

        self.object = WheeledRobot(
            prim_path=self.prim_path,
            name=cfg.robot_name,
            wheel_dof_names=cfg.wheel_dof_names,
            create_robot=True,
            usd_path=assets_root + cfg.usd_rel,
            position=self.initial_pos,
        )

        if cfg.controller not in _SUPPORTED_CONTROLLERS:
            raise RuntimeError(
                f"Robot controller {cfg.controller!r} not supported. "
                f"Add a branch for it in robot.py, or set "
                f"cfg.env.robot.controller to one of {sorted(_SUPPORTED_CONTROLLERS)}."
            )
        if cfg.controller == "Differential":
            self.controller = DifferentialController(
                name="diff_controller",
                wheel_radius=cfg.wheel_radius,
                wheel_base=cfg.wheel_base,
            )

    def reset(self):
        """Function to reset the robots velocities and position."""
        self.object.set_world_pose(position=self.initial_pos, orientation=self.initial_ori)
        self.object.set_linear_velocity(np.zeros(3))
        self.object.set_angular_velocity(np.zeros(3))

    def get_world_pose(self, ori_to_yaw=True):
        """
        Returns the current pose of the robot in the world frame.
        
        This function utilized a translation form quaternions to yaw.
        The former is what isaac sim provides and the latter what our model needs.
        """
        pos, ori = self.object.get_world_pose()
        if not(ori_to_yaw):
            return pos, ori
        # quaternion -> yaw
        w, x, y, z = ori
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return pos, yaw

    def apply_action(self, action):
        """Applies the given action on the robot."""
        v, w = action
        commands = self.controller.forward(np.array([float(v), float(w)]))
        self.object.apply_wheel_actions(commands)
