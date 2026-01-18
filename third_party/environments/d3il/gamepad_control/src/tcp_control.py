import abc

import numpy as np

from d3il_sim.sims.universal_sim.PrimitiveObjects import Box
from d3il_sim.utils import geometric_transformation as geo_trans

from .devices import gamepad, phyphox


class InteractiveTcpControllerBase(abc.ABC):
    def __init__(self, scene, robot, absolute_orientation=False):
        self.quat_operation = lambda x, y: geo_trans.quat_mul(x, y)
        if absolute_orientation:
            self.quat_operation = lambda x, y: x + y

        self.scene = scene
        scene.add_object(
            Box(
                "tcp_indicator",
                [0.55, 0.0, 0.7],
                [0, 1, 0, 0],
                static=True,
                visual_only=True,
            )
        )
        self.robot = robot

    def move(self, fix_xy_rot=True, timesteps=35):
        tcp_pos = self.scene.get_obj_pos(obj_name="tcp_indicator")
        tcp_quat = self.scene.get_obj_quat(obj_name="tcp_indicator")

        target_pos, target_quat = self.move_cmd(tcp_pos, tcp_quat)

        if fix_xy_rot:
            target_quat = [0, 1, 0, 0]

        target_pos, target_quat = self.robot._localize_cart_coords(
            target_pos, target_quat
        )

        self.robot.cartesianPosQuatTrackingController.setSetPoint(
            np.hstack((target_pos, target_quat))
        )
        self.robot.cartesianPosQuatTrackingController.executeControllerTimeSteps(
            self.robot, timesteps, block=False
        )

        for i in range(timesteps):
            self.scene.set_obj_pos(target_pos, obj_name="tcp_indicator")
            self.scene.set_obj_quat(target_quat, obj_name="tcp_indicator")
            self.scene.next_step()

        return target_pos, target_quat, self.robot.des_joint_pos

    def reset_pose(self):
        self.scene.reset()
        self.robot.beam_to_joint_pos(self.robot.init_qpos)
        self.scene.set_obj_pos(
            self.robot.current_c_pos_global, obj_name="tcp_indicator"
        )
        self.scene.set_obj_quat(
            self.robot.current_c_quat_global, obj_name="tcp_indicator"
        )

    def move_cmd(self, tcp_pos, tcp_quat):
        new_pos = tcp_pos + self.read_ctrl_pos()
        new_quat = self.quat_operation(tcp_quat, self.read_ctrl_quat())
        return new_pos, new_quat

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def read_ctrl_pos(self):
        pass

    @abc.abstractmethod
    def read_ctrl_quat(self):
        pass


class TcpGamepadController(InteractiveTcpControllerBase):
    def __init__(self, scene, robot, absolute_orientation=False):
        super().__init__(scene, robot, absolute_orientation)
        self.ctrl_device = gamepad.GamePad()
        self.rot_dampening = 20.0
        self.pos_dampening = 120.0

        if absolute_orientation:
            self.rot_dampening = 1.0

    def btn_a(self):
        return self.ctrl_device.BTN_SOUTH == 1

    def btn_b(self):
        return self.ctrl_device.BTN_EAST == 1

    def btn_x(self):
        return self.ctrl_device.BTN_NORTH == 1

    def btn_y(self):
        return self.ctrl_device.BTN_WEST == 1

    def reset(self):
        return self.ctrl_device.BTN_THUMBL == 1

    def start(self):
        return self.ctrl_device.BTN_START == 1

    def select(self):
        return self.ctrl_device.BTN_SELECT == 1

    def read_ctrl_pos(self):
        pos_dir = (
            np.array(
                [
                    self.ctrl_device.ABS_Y,
                    self.ctrl_device.ABS_X,
                    self.ctrl_device.BTN_TR - self.ctrl_device.BTN_TL,
                ]
            )
            / self.pos_dampening
        )

        pos_dir[np.abs(pos_dir) < 0.005] = 0.0
        return pos_dir

    def read_ctrl_quat(self):
        euler = (
            np.array(
                [
                    self.ctrl_device.ABS_RX * np.pi,
                    self.ctrl_device.ABS_RY * np.pi,
                    (self.ctrl_device.ABS_RZ - self.ctrl_device.ABS_Z) * np.pi,
                ]
            )
            / self.rot_dampening
        )

        euler[np.abs(euler) < 0.01] = 0.0

        return geo_trans.euler2quat(euler)


class TcpPhyPhoxController(InteractiveTcpControllerBase):
    def __init__(self, scene, robot, url):
        super().__init__(scene, robot, True)
        self.ctrl_device = phyphox.PhyPhoxIMU(url)

    def reset(self):
        return False

    def read_ctrl_pos(self):
        return (
            np.array(
                [self.ctrl_device.xPos, self.ctrl_device.yPos, self.ctrl_device.zPos]
            )
            / 1.0
        )

    def read_ctrl_quat(self):
        x, y, z = self.ctrl_device.xRot, self.ctrl_device.yRot, self.ctrl_device.zRot

        return geo_trans.euler2quat(np.array([x, y, z]) / 1.0)


class TcpComboController(TcpGamepadController):
    def __init__(self, scene, robot, url):
        super().__init__(scene, robot, True)
        self.rot_ctrl = TcpPhyPhoxController(url)

    def read_ctrl_quat(self):
        return self.rot_ctrl.read_ctrl_quat()
