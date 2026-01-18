import abc

import gym
import numpy as np

import environments.d3il.d3il_sim.controllers.Controller as ctrl
from environments.d3il.d3il_sim.controllers import CartPosQuatImpedenceController
from environments.d3il.d3il_sim.controllers.TrajectoryTracking import (
    CartPosQuatTrajectoryTracker,
    GotoCartPosQuatImpedanceController,
    GotoTrajectoryLinear,
)


class GymController(abc.ABC):
    def __init__(self, robot, controller):
        self.controller = controller
        self.robot = robot

    @abc.abstractmethod
    def action_space(self):
        raise NotImplementedError

    def set_action(self, action):
        pass

    def reset(self):
        self.controller.reset()

    def execute_action(self, n_time_steps):
        self.controller.executeControllerTimeSteps(self.robot, timeSteps=n_time_steps)


class GymJointPosController(GymController):
    def __init__(self, robot):

        super().__init__(robot, ctrl.JointPositionController())

    def action_space(self):
        # upper and lower bound for each joint position given by
        # https://frankaemika.github.io/docs/control_parameters.html#controller-requirements
        low = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        high = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

        action_space = gym.spaces.Box(low=np.array(low), high=np.array(high))

        return action_space

    def set_action(self, action):
        self.controller.setSetPoint(desired_pos=action)


class GymJointVelController(GymController):
    def __init__(self, robot):

        super().__init__(robot, ctrl.JointVelocityController())

    def action_space(self):
        # upper and lower bound for each joint position given by
        # https://frankaemika.github.io/docs/control_parameters.html#controller-requirements
        low = [-2.175, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100]
        high = [2.175, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]

        action_space = gym.spaces.Box(low=np.array(low), high=np.array(high))

        return action_space

    def set_action(self, action):
        self.controller.setSetPoint(desired_vel=action)


class GymTorqueController(GymController):
    def __init__(self, robot):
        super().__init__(robot, ctrl.TorqueController())

    def action_space(self):
        # upper and lower bound for each joint position given by
        # https://frankaemika.github.io/docs/control_parameters.html#controller-requirements
        low = [-2.175, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100]
        high = [2.175, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]

        action_space = gym.spaces.Box(low=np.array(low), high=np.array(high))

        return action_space

    def set_action(self, action):
        self.controller.setAction(action)


class GymCartesianVelController(GymController):
    def __init__(self, robot, fixed_orientation=None, max_cart_vel=0.05):
        controller = CartPosQuatImpedenceController()
        super(GymCartesianVelController, self).__init__(robot, controller)
        self.fixed_orientation = fixed_orientation
        self.max_cart_vel = max_cart_vel
        self.max_cart_pos = np.array([0.7, 0.4, 0.6])
        self.min_cart_pos = np.array([0.2, -0.4, 0.0])
        self.desired_c_pos = robot.des_c_pos

    def action_space(self):
        # upper and lower bound on the actions
        low_xyz_gripper = np.array([-1.0, -1.0, -1.0]) * self.max_cart_vel
        high_xyz_gripper = np.array([1.0, 1.0, 1.0]) * self.max_cart_vel

        quat_low_wxyz = np.array([-1, -1, -1, -1])
        quat_high_wxyz = np.array([1, 1, 1, 1])

        low_limits = low_xyz_gripper
        high_limits = high_xyz_gripper

        if self.fixed_orientation is None:
            low_limits.append(quat_low_wxyz)
            high_limits.append(quat_high_wxyz)

        action_space = gym.spaces.Box(low=low_limits, high=high_limits)
        return action_space

    def set_action(self, action):
# <<<<<<< HEAD
        # self.desired_c_pos = np.array(self.robot.current_c_pos) + np.array(action[:3])
        self.desired_c_pos = np.array(action[:3])
        # self.desired_c_pos = np.clip(
        #   self.desired_c_pos, self.min_cart_pos, self.max_cart_pos
        # )
# =======
#         self.desired_c_pos = np.array(self.robot.des_c_pos) + np.array(action[:3])
#         self.desired_c_pos = np.clip(
#             self.desired_c_pos, self.min_cart_pos, self.max_cart_pos
#         )
# >>>>>>> origin/controller_fixes

        if self.fixed_orientation is None:
            desired_quat = np.array(action[3:])
        else:
            desired_quat = self.fixed_orientation

        self.controller.setDesiredPos(
            np.concatenate([self.desired_c_pos, desired_quat])
        )


class GymXYVelController(GymController):
    def __init__(
        self, robot, fixed_orientation=None, fixed_z_pos=0.4, max_cart_vel=0.05
    ):
        controller = CartPosQuatImpedenceController()
        super(GymCartesianVelController, self).__init__(robot, controller)
        self.fixed_orientation = fixed_orientation
        self.fixed_z_pos = fixed_z_pos
        self.max_cart_vel = max_cart_vel
        self.max_cart_pos = np.array([0.7, 0.4])
        self.min_cart_pos = np.array([0.2, -0.4])
        self.desired_c_pos = robot.des_c_pos

    def action_space(self):
        # upper and lower bound on the actions
        low_xyz_gripper = np.array([-1.0, -1.0]) * self.max_cart_vel
        high_xyz_gripper = np.array([1.0, 1.0]) * self.max_cart_vel

        quat_low_wxyz = np.array([-1, -1, -1, -1])
        quat_high_wxyz = np.array([1, 1, 1, 1])

        low_limits = low_xyz_gripper
        high_limits = high_xyz_gripper

        if self.fixed_orientation is None:
            low_limits.append(quat_low_wxyz)
            high_limits.append(quat_high_wxyz)

        action_space = gym.spaces.Box(low=low_limits, high=high_limits)
        return action_space

    def set_action(self, action):
        self.desired_c_pos = np.array(self.robot.des_c_pos) + np.array(action[:2])
        self.desired_c_pos = np.clip(
            self.desired_c_pos, self.min_cart_pos, self.max_cart_pos
        )

        if self.fixed_orientation is None:
            desired_quat = np.array(action[2:])
        else:
            desired_quat = self.fixed_orientation

        self.controller.setDesiredPos(
            np.concatenate([self.desired_c_pos, desired_quat])
        )
