import numpy as np

from environments.d3il.d3il_sim.controllers.Controller import JointPDController
from environments.d3il.d3il_sim.controllers.IKControllers import CartPosQuatImpedenceController


class TeleopController(JointPDController):
    def __init__(self, goal_pos_fct, goal_vel_fct, goal_gripper_width_fct):
        super().__init__()
        self.goal_pos_fct = goal_pos_fct
        self.goal_vel_fct = goal_vel_fct
        self.goal_gripper_width_fct = goal_gripper_width_fct

    def getControl(self, robot):
        self.setSetPoint(
            desired_pos=self.goal_pos_fct(), desired_vel=self.goal_vel_fct()
        )
        if self.goal_gripper_width_fct() <= 0.003:
            robot.set_gripper_cmd_type = 1  # Grasp
        else:
            robot.set_gripper_cmd_type = 2  # Move
        robot.set_gripper_width = self.goal_gripper_width_fct()
        return super().getControl(robot)


class CartesianTeleopController(CartPosQuatImpedenceController):
    def __init__(self, target_cart_pos_fn, target_cart_quat_fn=None, height=0.3):
        super().__init__()
        self.target_cart_pos_fn = target_cart_pos_fn
        self.target_cart_quat_fn = target_cart_quat_fn
        self.height = height

    def getControl(self, robot):
        target_pos = self.target_cart_pos_fn()

        target_pos[2] = self.height

        target_quat = [0, 1, 0, 0]
        if self.target_cart_quat_fn is not None:
            target_quat = self.target_cart_quat_fn()

        target = np.hstack((target_pos, target_quat))
        self.setSetPoint(target)
        return super().getControl(robot)
