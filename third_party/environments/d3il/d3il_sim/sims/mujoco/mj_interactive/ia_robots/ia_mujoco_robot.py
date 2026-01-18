import time

import numpy as np

from environments.d3il.d3il_sim.core import logger
from environments.d3il.d3il_sim.sims.mujoco.MujocoRobot import MujocoRobot
from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box
from environments.d3il.d3il_sim.utils.sim_path import d3il_path


class InteractiveMujocoRobot(MujocoRobot):
    def __init__(
        self, scene, ia_tcp_ctrl, fix_xy_rot=True, recording_file=None, *args, **kwargs
    ):
        """
        Wrapper for the MujocoRobot.
        Handles the user interaction.

        :param scene: MujocoScene
        :param ia_tcp_ctrl: Interactive TCP Controller
        :param recording_file: File to save recorded poses to. If None, no file will be written
        :param args: MujocoRobot args
        :param kwargs: MujocoRobot kwargs
        """
        super(InteractiveMujocoRobot, self).__init__(scene, *args, **kwargs)

        self.interact_controller = ia_tcp_ctrl
        self.fix_xy_rot = fix_xy_rot

        self.pose_log = PoseLog(recording_file)
        self.is_init = False

        self.init_qpos = self.get_init_qpos()
        self.init_tcp_pos = [0.55, 0.0, 0.7]
        self.init_tcp_quat = [0, 1, 0, 0]

        scene.add_object(
            Box(
                "tcp_indicator",
                [0.55, 0.0, 0.7],
                [0, 1, 0, 0],
                static=True,
                visual_only=True,
            )
        )

    @property
    def xml_file_path(self):
        return d3il_path("./models/mujoco/robots/panda.xml")

    def _read_tcp(self):
        """
        Internal function to quickly access TCP Pos and Quat
        :return: tcp position, tcp orientation quaternion
        """
        return self.current_c_pos, self.current_c_quat

    def plot(self):
        """
        create plots. used to react to user input
        """
        self.stop_logging()
        self.robot_logger.plot(plot_selection=logger.RobotPlotFlags.JOINTS)

    def start(self):
        self.activeController = self.gotoCartPosQuatImpedanceController
        self.gotoCartPositionAndQuat_ImpedanceCtrl(
            self.init_tcp_pos, self.init_tcp_quat, block=False
        )

    def run(self):
        """
        Main Loop for controlling the Robot.

        Polls the controller for user input, especially for stopping.
        Calls internal movement_loop() function to handle movement
        """
        self.start()

        while True:
            # On Stop
            if self.interact_controller.stop():
                self.pose_log.write()
                break

            # On Move
            pos, quat, joints = self.movement_loop()

            # On Plot
            if self.interact_controller.plot():
                self.plot()

            # On Save Pose
            if self.interact_controller.save():
                self.pose_log.log(pos, quat, joints)

    def movement_loop(self):
        """
        One Iteration of the Movement Loop.
        Polls the Controller for desired position, computes the IK and drives to that configuration.
        :return: cartesian target position, cartesian target orientation, joint configuration
        """
        # Update State
        self.receiveState()
        tcp_pos, tcp_quat = self._read_tcp()

        tcp_pos = self.scene.get_obj_pos(obj_name="tcp_indicator")
        tcp_quat = self.scene.get_obj_quat(obj_name="tcp_indicator")

        # On Reset
        if self.interact_controller.reset():
            target_pos, target_quat, target_gripper = self.reset_pose()
        # On Move
        else:
            target_pos, target_quat = self.interact_controller.move(
                tcp_pos, tcp_quat, self.fix_xy_rot
            )
            target_gripper = self.interact_controller.grip(self.gripper_width)

        if self.fix_xy_rot:
            target_quat = [0, 1, 0, 0]
        self.open_fingers()

        self.cartesianPosQuatTrackingController.setSetPoint(
            np.hstack((target_pos, target_quat))
        )
        self.cartesianPosQuatTrackingController.executeControllerTimeSteps(
            self, 30, block=False
        )

        for i in range(30):
            self.scene.set_obj_pos(target_pos, obj_name="tcp_indicator")
            self.scene.set_obj_quat(target_quat, obj_name="tcp_indicator")
            self.nextStep()
        return target_pos, target_quat, self.des_joint_pos

    def beam_to_joint_pos(self, desiredJoints, run=True):
        super().beam_to_joint_pos(desiredJoints, run)
        self.scene.set_obj_pos(self.current_c_pos_global, obj_name="tcp_indicator")
        self.scene.set_obj_quat(self.current_c_quat_global, obj_name="tcp_indicator")

    def reset_pose(self):
        """
        resets the robot configuration to initial positions, including helper bot
        :return: cartesian target position, cartesian target orientation, joint configuration
        """
        self.scene.reset()
        self.beam_to_joint_pos(self.init_qpos)

        return self.init_tcp_pos, self.init_tcp_quat, 0.0


class PoseLog:
    def __init__(self, save_file=None):
        """
        Class for recording Poses, consisting of Cartesian Space Coordinates, Orientation
        and associated Joint Configuration

        :param save_file: Save file location. If None, no contents will be saved to disk.
        """
        self.file = save_file
        self._log = []
        self.last_ts = 0

    def log(self, pos, quat, joints):
        """
        save a pose to log
        :param pos: cartesian position
        :param quat: cartesian orientation
        :param joints: joint configuration
        :return: None
        """
        t = time.time()
        if t > self.last_ts + 1:
            self._log.append([pos, quat, joints])
            self.last_ts = t

    def write(self):
        """
        writes the pose log to the file.
        """
        if self.file is not None:
            arr = np.asarray(self._log)
            np.save(self.file, arr)
