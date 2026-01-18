import logging
import os
import time

import numpy as np
import py_at_broker as pab

from environments.d3il.d3il_sim.core import RobotBase, RobotDynamicsInterface


class SLDynamics(RobotDynamicsInterface):
    def __init__(self, robot):
        self.robot = robot

    def get_gravity(self, q=None):
        return self.robot.gravtiy

    def get_coriolis(self, q=None, qd=None):
        return self.robot.coriolis

    def get_mass_matrix(self, q=None):
        return self.robot.massMatrix


class SlRobot(RobotBase):
    def __init__(
        self,
        scene,
        robot_name: str = "panda2",
        num_DoF: int = 7,
        backend_addr: str = "tcp://localhost:51468",
        local_addr: str = None,
        gripper_actuation: bool = False,
    ):
        """Creates an SLRobot.

        Args:
            robot_name (str, optional): the name of the physical robot. MUST be the same as compiled into SL. Defaults to "panda2".
            gin_path ([type], optional): path to the gin configuration files. If None, defaults will be used. Defaults to None.
            config_path (str, optional): path to the legacy config directory. If None, defaults will be used. Defaults to None.
            num_DoF (int, optional): Number of Degrees of Freedom in the Robot. Defaults to 7.
            backend_addr (str, optional): Address of the SL Control PC. Must be changed if you are controlling a robot connected on a different PC.
                Defaults to "tcp://localhost:51468".
            local_addr (str, optional): If backend_addr is changed to control a remote robot, you must define the address of the PC running the python script.
                Defaults to None.
            gripper_actuation (bool, optional): Gripper actuation. Defaults to False.
        """

        super(SlRobot, self).__init__(scene, 0.001, num_DoF=num_DoF)

        self.dynModel = SLDynamics(self)
        # Set RT Scheduler
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(35))
        except OSError as err:
            logging.getLogger(__name__).error(
                "Error: Failed to set proc. to real-time scheduler\n{0}".format(err)
            )

        # Create Network
        self.robot_name = robot_name
        if local_addr:
            self.broker = pab.broker(backend_addr=backend_addr, local_addr=local_addr)
        else:
            self.broker = pab.broker(backend_addr=backend_addr)

        self.broker.register_signal(self.robot_name + "_des_tau", pab.MsgType.des_tau)
        self.broker.register_signal(
            self.robot_name + "_gripper", pab.MsgType.gripper_cmd
        )
        self.broker.request_signal(
            self.robot_name + "_state", pab.MsgType.franka_state, True
        )

        self.gripper_actuation = gripper_actuation
        self.set_gripper_cmd_type = 2
        self.use_inv_dyn = False
        self.clip_rate = True

        self.frame_number = 0
        self.is_real_robot = False

        self.rate_limit = 0.8  # Rate torque limit
        self.max_sync_jitter = 0.2
        self.torque_limit = np.array(
            [40, 40, 40, 40, 10, 10, 10], dtype=np.float64
        )  # Absolute torque limit
        self.send_command_flag = True

        self._home_q_pos = np.array(
            [
                3.46311499e-05,
                9.08202901e-02,
                -7.06741586e-04,
                -1.58921993e00,
                -1.13537759e-02,
                1.90483785e00,
                7.85356522e-01,
            ]
        )

        self.reset()
        self.receiveState()

    """
    def getJacobian(self, q=None, linkID=6):
        if q is None:
            q = self.current_j_pos

        tmp, Jac = sl.Jacobian(q.reshape((1, q.shape[0])), linkID)

        return Jac

    def getForwardKinematics(self, q=None):
        if q is None:
            q = self.current_j_pos

        tmp, fw = sl.FK(q.reshape((1, q.shape[0])))
        fw = fw.reshape((self.num_DoF - 1, -1))

        cart = fw[:, 0:3]
        euler = fw[:, 3:6]
        quat = fw[:, 6:]

        return cart, quat, euler
    """

    def receiveState(self):

        # Receive next state message
        recv_start = time.clock_gettime(time.CLOCK_MONOTONIC)

        # Receiving state and desired control messages
        # Wait for a new policy message and do control with the latest state msg available.
        # For the first time or in case of com. lost wait for the robot to init (i.e. publish a msg)
        if self.step_count <= 1:
            msg_panda = self.broker.recv_msg(self.robot_name + "_state", -1)

        msg_panda = self.broker.recv_msg(self.robot_name + "_state", 0)
        repeatReceive = msg_panda.get_fnumber() == self.frame_number

        while repeatReceive:
            msg_panda = self.broker.recv_msg(self.robot_name + "_state", 0)
            repeatReceive = (
                msg_panda.get_fnumber() == self.frame_number and self.send_command_flag
            )

        recv_stop = time.clock_gettime(time.CLOCK_MONOTONIC)

        # Catch no robot state message for long time
        # if (2 * recv_stop - recv_start - msg_panda.get_timestamp()) > self.max_sync_jitter:
        #   logging.getLogger(__name__).info("De-synced, messages too old\n Resetting...")

        self.current_j_vel = msg_panda.get_j_vel()
        self.current_j_pos = msg_panda.get_j_pos()

        # TODO: Forward Kinematics of SL seems to be different then the one from pinochio. We need to check why this
        #  is the case

        # self.current_c_pos = msg_panda.get_c_pos()
        # self.current_c_quat = msg_panda.get_c_ori_quat()
        self.current_c_pos, self.current_c_quat = self.getForwardKinematics()

        self.current_c_vel = msg_panda.get_c_vel()
        self.current_c_quat_vel = msg_panda.get_dc_ori_quat()

        self.gripper_width = msg_panda.get_gripper_state()
        self.tau_ext_hat_filtered = msg_panda.get_tau_ext_hat_filtered()

        self.last_cmd = msg_panda.get_last_cmd()
        self.current_load = msg_panda.get_j_load()

        if msg_panda.get_flag_real_robot():
            self.time_keeper.time_stamp = msg_panda.get_timestamp()
        else:
            self.time_keeper.time_stamp = msg_panda.get_fnumber() / 1000

        self.frame_number = msg_panda.get_fnumber()
        self.is_real_robot = msg_panda.get_flag_real_robot()

        #        logging.getLogger(__name__).debug('{}:'.format(self.frame_number))
        #        logging.getLogger(__name__).debug(self.grav_cmp)
        #        logging.getLogger(__name__).debug(self.last_cmd)

        self.massMatrix = msg_panda.get_mass().reshape(msg_panda.get_mass_dim())
        self.gravity = msg_panda.get_gravity()
        self.coriolis = msg_panda.get_coriolis()

        self.o_t_ee_Matrix = msg_panda.get_o_t_ee().reshape(4, 4)

        self.send_command_flag = False
        # FIXME: Set Fingers to Zero
        # self.current_fing_pos = np.array([[0], [0]])
        # self.current_fing_vel = np.array([[0], [0]])

    def prepare_step(self):
        # Send command. The default behavior is to have an activeController. The absence of it is only used in the old
        # teleop setup (April 2022), where the manual control is given directly.
        if self.activeController is not None:
            self.command = self.activeController.getControl(self)
        self.preprocessCommand(self.command)
        msg_out = pab.des_tau_msg()
        msg_out.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
        msg_out.set_fnumber(self.frame_number)
        msg_out.set_j_torque_des(self.uff)
        self.broker.send_msg(self.robot_name + "_des_tau", msg_out)
        if self.gripper_actuation:
            gripper_msg_out = pab.gripper_cmd_msg()
            gripper_msg_out.set_timestamp(time.clock_gettime(time.CLOCK_MONOTONIC))
            gripper_msg_out.set_fnumber(self.frame_number)
            gripper_msg_out.set_cmd_t(
                self.set_gripper_cmd_type
            )  # This is the enum number for Moving
            if self.set_gripper_cmd_type == 2:
                gripper_msg_out.set_width(self.set_gripper_width)
                gripper_msg_out.set_speed(0.1)
            elif self.set_gripper_cmd_type == 1:
                gripper_msg_out.set_width(self.set_gripper_width)
                gripper_msg_out.set_speed(0.1)
                gripper_msg_out.set_force(10)
                gripper_msg_out.set_epsilon_out(0.1)
                gripper_msg_out.set_epsilon_in(0.1)
            self.broker.send_msg(self.robot_name + "_gripper", gripper_msg_out)

        self.send_command_flag = True
        self.receiveState()

    def preprocessCommand(self, target_j_acc):

        if len(target_j_acc.shape) == 2:
            target_j_acc = target_j_acc.reshape((-1))

        if target_j_acc.shape[0] != self.num_DoF:
            raise ValueError(
                "Specified motor command vector needs to be of size %d\n",
                target_j_acc.shape[0],
            )

        if self.use_inv_dyn:
            self.des_joint_acc = target_j_acc
            self.uff = self.massMatrix.dot(target_j_acc) + self.coriolis

            """res, tmp_grav = GravCmp(np.array([np.concatenate([self.current_j_pos, self.current_j_vel])]))
            assert res == 1
            res, tmp_corr = IDyn(np.array([np.concatenate([self.current_j_pos, self.current_j_vel])]),
                                    np.array([np.concatenate([self.current_j_pos, self.current_j_vel, np.zeros((7,))])]))
            assert res == 1

            coriolis_temp = tmp_corr - tmp_grav
            logging.getLogger(__name__).debug(coriolis_temp - self.coriolis)"""

            # self.uff -= self.grav_cmp
            # self.uff = self.uff.reshape((-1,))

        else:
            self.uff = target_j_acc

        # Catch start of control
        if self.step_count == 0:
            self.uff_last = 0

        if self.clip_rate:
            # Clip and rate limit torque
            uff_diff = self.uff - self.uff_last
            uff_diff = np.clip(uff_diff, -self.rate_limit, self.rate_limit)
            self.uff = self.uff_last + uff_diff

        self.uff = np.clip(self.uff, -self.torque_limit, self.torque_limit)
        self.uff_last = self.uff

    def get_command_from_inverse_dynamics(
        self, target_j_acc, mj_calc_inv=False, robot_id=None, client_id=None
    ):
        pass

    def beam_to_cart_pos_and_quat(self, desiredPos, desiredQuat):
        pass

    def beam_to_joint_pos(self, desiredJoints):
        pass

    def set_q(self, joints, robot_id=None, physicsClientId=None):
        pass

    def set_home(self, q_pos):
        self._home_q_pos = q_pos

    def go_home(self, duration=4):
        self.gotoJointPosition(self._home_q_pos, duration=duration)
