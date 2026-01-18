import numpy as np

from environments.d3il.d3il_sim.controllers.Controller import ControllerBase
from environments.d3il.d3il_sim.sims.sl.multibot_teleop.src.kalman_filter import KalmanFilter
from environments.d3il.d3il_sim.sims.sl.SlRobot import SlRobot


class HumanController(ControllerBase):
    def __init__(self, primary_robot, regularize=True):
        super().__init__()
        # gains
        # old gains: #  0.7 * np.array([1.8, 0.2, 1.0, 1.3, 1.0, 1.0, 1.5])
        # how much the external forces should act on the robot
        self.gain = np.array([0.26, 0.44, 0.40, 1.11, 1.10, 1.20, 0.85])
        # how much the min/max joint positions should act on the joints: pushes into the center

        if regularize:
            self.reg_gain = np.array([5.0, 2.2, 1.3, 0.9, 0.1, 0.1, 0.0])
        else:
            self.reg_gain = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.extra_anti_gravity_gain = 0.0
        # kalman filter for smoothing
        self._primary_load_filter = KalmanFilter(self.get_raw_load(primary_robot))
        self.primary_robot = primary_robot

    def get_raw_load(self, robot):
        return robot.tau_ext_hat_filtered - self.extra_anti_gravity_gain * robot.gravity

    def get_regularization_load(self, robot: SlRobot):
        left_boundary = 1 / np.clip(
            np.abs(robot.joint_pos_min - robot.current_j_pos), 1e-8, 100000
        )
        right_boundary = 1 / np.clip(
            np.abs(robot.joint_pos_max - robot.current_j_pos), 1e-8, 100000
        )
        return -right_boundary + left_boundary

    def getControl(self, robot):
        assert robot is self.primary_robot
        human_torque = self.gain * (
            -self._primary_load_filter.get_filtered(self.get_raw_load(robot))
        )
        regularization_torque = self.reg_gain * self.get_regularization_load(robot)

        return human_torque + regularization_torque
