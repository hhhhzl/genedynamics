"""
This module contains the inverse kinematics controller.
"""

from abc import abstractmethod
import logging

import numpy as np

import environments.d3il.d3il_sim.controllers.GainsInterface as gains
import environments.d3il.d3il_sim.utils as utils
from environments.d3il.d3il_sim.controllers.Controller import TrackingController


class CartPosImpedenceController(TrackingController, gains.CartPosControllerConfig):
    """
    Controller for the cartesian coordinates of the robots end effector.
    """

    def __init__(self):

        TrackingController.__init__(self, dimSetPoint=3)
        gains.CartPosControllerConfig.__init__(self)

        self.J_reg = 1e-6  # Jacobian regularization constant
        self.W = np.diag([1, 1, 1, 1, 1, 1, 1])

        # Null-space theta configuration
        self.target_th_null = np.array(
            [
                3.57795216e-09,
                1.74532920e-01,
                3.30500960e-08,
                -8.72664630e-01,
                -1.14096181e-07,
                1.22173047e00,
                7.85398126e-01,
            ]
        )

        self.reset()

    def reset(self):
        self.desired_c_pos = np.array([0.624, 0, 0.55])
        self.desired_c_vel = np.zeros((3,))
        self.desired_c_acc = np.zeros((3,))

    def getControl(self, robot):
        """
        Calculates the robot joint acceleration based on
        - the current joint velocity
        - the current joint positions

        :param robot: instance of the robot
        :return: target joint acceleration (num_joints, )
        """
        self.paramsLock.acquire()
        xd_d = self.desired_c_pos - robot.current_c_pos
        target_c_acc = self.pgain * xd_d

        J = robot.getJacobian()
        J = J[:3, :]
        Jw = J.dot(self.W)

        # J *  W * J' + reg * I
        JwJ_reg = Jw.dot(J.T) + self.J_reg * np.eye(3)

        # Null space movement
        qd_null = self.pgain_null * (self.target_th_null - robot.current_j_pos)
        # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null

        qd_d = np.linalg.solve(JwJ_reg, target_c_acc - J.dot(qd_null))
        qd_d = self.W.dot(J.transpose()).dot(qd_d) + qd_null

        robot.des_c_pos = self.desired_c_pos
        robot.des_c_vel = self.desired_c_vel

        robot.jointTrackingController.setSetPoint(robot.current_j_pos, qd_d)
        self.paramsLock.release()

        return robot.jointTrackingController.getControl(robot)

    def setGains(self, pGain, dGain):
        """
        Setter for the gains of the PD Controller.

        :param pGain: p gain
        :param dGain: d gain
        :return: no return value
        """
        self.paramsLock.acquire()
        self.pgain = pGain
        self.dgain = dGain
        self.paramsLock.release()

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
        """
        Sets the desired position, velocity and acceleration of the joints.

        :param desired_pos: desired position (num_joints,)
        :param desired_vel: desired velocity (num_joints,)
        :param desired_acc: desired acceleration (num_joints,)
        :return: no return value
        """
        self.paramsLock.acquire()
        self.desired_c_pos = desired_pos
        if desired_vel is not None:
            self.desired_c_vel = desired_vel
        if desired_acc is not None:
            self.desired_c_acc = desired_acc
        self.paramsLock.release()

    def getCurrentPos(self, robot):
        """
        Getter for the robots current posi
        :param robot:
        :return:
        """
        return robot.current_c_pos

    def getDesiredPos(self, robot):
        return robot.des_c_pos


class CartesianPositionController(CartPosImpedenceController):
    def setAction(self, action):
        self.desired_c_pos = action

    @abstractmethod
    def reset(self):
        pass


class CartPosQuatImpedenceController(
    TrackingController, gains.CartPosQuatControllerConfig
):
    """
    Controller for the cartesian coordinates and the orientation (using quaternions) of the robots end effector.
    """

    def __init__(self):

        TrackingController.__init__(self, dimSetPoint=7)
        gains.CartPosQuatControllerConfig.__init__(self)
        self.W = np.diag(self.W)
        self.dgain_null = np.sqrt(self.pgain_null) * 2

        self.reset()
        # with this flag set on, the robot is always beamed to the correct location during the IK. Mainly used for debugging reasons (just test IK without dynamics)
        self.neglect_dynamics = False

    def reset(self):
        self.desired_c_pos = np.array([0.624, 0, 0.55])
        self.desired_quat = np.array([0.0, 0.984, 0, 0.177])
        self.desired_quat = self.desired_quat / np.linalg.norm(self.desired_quat)
        self.desired_c_vel = np.zeros((3,))
        self.desired_quat_vel = np.zeros((4,))

        self.old_des_joint_vel = np.zeros((7,))
        self.old_q = np.zeros((7,))
        self.old_q[:] = np.nan

    def getControl(self, robot):

        self.paramsLock.acquire()
        super(CartPosQuatImpedenceController, self).getControl(robot)

        if any(np.isnan(self.old_q)):
            self.old_q = robot.current_j_pos.copy()

        q = self.old_q.copy()

        q = (
            self.joint_filter_coefficient * q
            + (1 - self.joint_filter_coefficient) * robot.current_j_pos
        )

        qd_dsum = np.zeros(q.shape)

        oldErrorNorm = np.inf
        qd_d = np.zeros(q.shape)

        [current_c_pos, current_c_quat] = robot.getForwardKinematics(q)
        target_cpos_acc = self.desired_c_pos - current_c_pos

        curr_quat = current_c_quat

        if np.linalg.norm(curr_quat - self.desired_quat) > np.linalg.norm(
            curr_quat + self.desired_quat
        ):
            curr_quat = -curr_quat
        oldErrorNorm = np.sum(target_cpos_acc**2) + np.sum(
            (curr_quat - self.desired_quat) ** 2
        )

        des_quat = self.desired_quat
        for i in range(self.num_iter):

            [current_c_pos, current_c_quat] = robot.getForwardKinematics(q)
            target_cpos_acc = self.desired_c_pos - current_c_pos

            curr_quat = current_c_quat

            if np.linalg.norm(curr_quat - des_quat) > np.linalg.norm(
                curr_quat + des_quat
            ):
                des_quat = -des_quat
            errNorm = np.sum(target_cpos_acc**2) + np.sum((curr_quat - des_quat) ** 2)

            target_cquat = utils.get_quaternion_error(curr_quat, des_quat)

            target_cpos_acc = np.clip(target_cpos_acc, -0.01, 0.01)
            target_cquat = np.clip(target_cquat, -0.1, 0.1)
            # self.pgain_quat = np.zeros((3,))

            target_c_acc = np.hstack(
                (self.pgain_pos * target_cpos_acc, self.pgain_quat * target_cquat)
            )

            J = robot.getJacobian(q)

            # Singular Value decomposition, to clip the singular values which are too small/big

            Jw = J.dot(self.W)

            # J *  W * J' + reg * I
            condNumber = np.linalg.cond(Jw.dot(J.T))
            JwJ_reg = Jw.dot(J.T) + self.J_reg * np.eye(J.shape[0])

            u, s, v = np.linalg.svd(JwJ_reg, full_matrices=False)
            s_orig = s
            s = np.clip(s, self.min_svd_values, self.max_svd_values)
            # reconstruct the Jacobian
            JwJ_reg = u @ np.diag(s) @ v
            condNumber2 = np.linalg.cond(JwJ_reg)
            largestSV = np.max(s_orig)

            qdev_rest = np.clip(self.rest_posture - q, -0.2, 0.2)

            # Null space movement
            qd_null = np.array(
                self.pgain_null * (qdev_rest)
            )  # + self.dgain_null * (-robot.current_j_vel)

            margin_to_limit = 0.01
            pgain_limit = 20

            qd_null_limit = np.zeros(qd_null.shape)
            qd_null_limit_max = pgain_limit * (
                robot.joint_pos_max - margin_to_limit - q
            )
            qd_null_limit_min = pgain_limit * (
                robot.joint_pos_min + margin_to_limit - q
            )
            qd_null_limit[
                q > robot.joint_pos_max - margin_to_limit
            ] += qd_null_limit_max[q > robot.joint_pos_max - margin_to_limit]
            qd_null_limit[
                q < robot.joint_pos_min + margin_to_limit
            ] += qd_null_limit_min[q < robot.joint_pos_min + margin_to_limit]

            # qd_null += qd_null_limit

            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, target_c_acc - J.dot(qd_null))
            qd_d = self.W.dot(J.transpose()).dot(qd_d) + qd_null

            # clip desired joint velocities for stability

            if np.linalg.norm(qd_d) > 3:
                qd_d = qd_d * 3 / np.linalg.norm(qd_d)

            qd_dsum = qd_dsum + qd_d

            q = q + self.learningRate * qd_d
            q = np.clip(q, robot.joint_pos_min, robot.joint_pos_max)

        self.tracking_error = np.sum(np.abs(self.desired_c_pos - current_c_pos)) > 0.01

        qd_dsum = (q - self.old_q) / robot.dt
        des_acc = self.ddgain * (qd_dsum - self.old_des_joint_vel) / robot.dt

        if np.sum(np.abs(self.desired_c_pos - current_c_pos)) > 0.1:
            target_cquat = utils.get_quaternion_error(curr_quat, self.desired_quat)

            logging.getLogger(__name__).debug(
                "i: %d, Time: %f, Pos_error: %f, Quat_error: %f,  qd_d: %f"
                % (
                    i,
                    robot.time_stamp,
                    np.linalg.norm(self.desired_c_pos - current_c_pos),
                    np.linalg.norm(target_cquat),
                    np.linalg.norm(qd_d),
                ),
                qd_d,
                self.desired_c_pos,
                des_acc,
            )

        if np.linalg.norm(des_acc) > 10000:
            des_acc = des_acc * 10000 / np.linalg.norm(des_acc)

        robot.jointTrackingController.setSetPoint(q, qd_dsum, des_acc)
        # robot.jointTrackingController.setSetPoint(q, qd_dsum)#, des_acc)

        self.old_q = q.copy()
        self.old_des_joint_vel = qd_dsum

        robot.des_c_pos = self.desired_c_pos
        robot.des_c_vel = self.desired_c_vel
        robot.des_quat = self.desired_quat
        robot.des_quat_vel = self.desired_quat_vel
        robot.misc_data = np.array([errNorm, condNumber, condNumber2, largestSV])

        self.paramsLock.release()

        if self.neglect_dynamics:
            robot.beam_to_joint_pos(q, resetDesired=False)
            return np.zeros((7,))
        else:
            control = robot.jointTrackingController.getControl(robot)

            return control

    '''
    def setGains(self, pGain, dGain, pGain_null, dGain_null, dGainVelCtrl):
        """
        Setter for the gains of the PD Controller.

        :param pGain: p gain
        :param dGain: d gain
        :param pGain_null: p gain null
        :param dGain_null: d gain null
        :param dGainVelCtrl: gain for velocity control on top
        :return: no return value
        """
        # self.paramsLock.acquire()
        self.pgain = pGain
        self.dgain = dGain
        self.pgain_null = pGain_null
        self.dgain_null = dGain_null
        self.dgain_velcontroller = dGainVelCtrl
        # self.paramsLock.release()
    '''

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
        """
        Sets the desired position, velocity and acceleration of the joints.

        :param desired_pos: desired position (num_joints,)
        :param desired_vel: desired velocity (num_joints,)
        :param desired_acc: desired acceleration (num_joints,)
        :return: no return value
        """
        self.paramsLock.acquire()
        self.desired_c_pos = desired_pos[:3].copy()
        self.desired_quat = desired_pos[3:] / np.linalg.norm(desired_pos[3:])
        if desired_vel is not None:
            self.desired_c_vel = desired_vel[:3]
            self.desired_quat_vel = desired_vel[3:]

        self.paramsLock.release()

    def getCurrentPos(self, robot):
        """
        Getter for the robots current positions.

        :param robot: instance of the robot
        :return: current joint position (num_joints, 1)
        """
        return np.hstack((robot.current_c_pos, robot.current_c_quat))

    def getDesiredPos(self, robot):
        return np.concatenate((robot.des_c_pos, robot.des_quat))


class CartVelocityImpedenceController(CartPosQuatImpedenceController):
    """
    Controller for the cartesian coordinates and the orientation (using quaternions) of the robots end effector.
    """

    def __init__(self, fixed_orientation=None, max_cart_vel=0.5):

        super(CartVelocityImpedenceController, self).__init__()
        self.fixed_orientation = fixed_orientation
        self.max_cart_vel = max_cart_vel

        self.max_cart_pos = np.array([0.7, 0.4, 0.75])
        self.min_cart_pos = np.array([0.2, -0.4, 0.0])

    def getControl(self, robot):

        self.desired_c_pos = np.array(robot.current_c_pos) + robot.dt * np.array(
            self.desired_c_vel
        )
        self.desired_c_pos = np.clip(
            self.desired_c_pos, self.min_cart_pos, self.max_cart_pos
        )

        if self.fixed_orientation is None:
            self.desired_quat = self.fixed_orientation
        else:
            self.desired_quat = self.fixed_orientation

        return super(CartVelocityImpedenceController, self).getControl(robot)

    def setSetPoint(self, desired_pos, desired_vel, desired_acc=None):
        """
        Sets the desired position, velocity and acceleration of the joints.

        :param desired_pos: desired position (num_joints,)
        :param desired_vel: desired velocity (num_joints,)
        :param desired_acc: desired acceleration (num_joints,)
        :return: no return value
        """
        self.paramsLock.acquire()
        self.desired_c_vel = desired_vel[:3]
        if self.fixed_orientation is None:
            self.desired_quat_vel = desired_vel[3:]

        self.paramsLock.release()


class CartPosQuatImpedenceJacTransposeController(
    CartPosQuatImpedenceController, gains.CartPosQuatJacTransposeControllerConfig
):
    def __init__(self):
        super(CartPosQuatImpedenceJacTransposeController, self).__init__()
        gains.CartPosQuatJacTransposeControllerConfig.__init__(self)

    def getControl(self, robot):
        self.paramsLock.acquire()
        qd_d = self.desired_c_pos - robot.current_c_pos
        target_cpos_acc = self.pgain_pos * qd_d
        curr_quat = (
            -robot.current_c_quat
            if (robot.current_c_quat @ self.desired_quat) < 0
            else robot.current_c_quat
        )
        target_cquat = self.pgain_quat * utils.get_quaternion_error(
            robot.current_c_quat, self.desired_quat
        )
        J = robot.getJacobian()

        qd_null = self.pgain_null * (
            self.rest_posture - robot.current_j_pos
        ) + self.dgain_null * (-robot.current_j_vel)
        target_acc = np.concatenate((target_cpos_acc, target_cquat))
        qdd = J.T @ (target_acc - self.dgain * (J @ robot.current_j_vel)) + qd_null
        # qdd = J.T@(self.pgain*pos_error + self.dgain*vel_error) + qd_null
        robot.des_c_pos = self.desired_c_pos
        robot.des_c_vel = self.desired_c_vel
        robot.des_quat = self.desired_quat
        robot.des_quat_vel = self.desired_quat_vel
        self.paramsLock.release()
        return qdd


class CartPosQuatCartesianRobotController(TrackingController):
    def __init__(self):
        TrackingController.__init__(self, dimSetPoint=7)
        self.reset()
    def reset(self):
        self.desired_pos = np.zeros((self.dimSetPoint,))

    def setAction(self, action):
        self.desired_pos = action  # should be named with quad

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
        self.desired_pos = desired_pos

    def getControl(self, robot):
        robot.des_c_pos = self.desired_pos[:3]
        robot.des_c_vel = np.zeros((3,))
        if self.desired_pos.shape[0] > 3:
            robot.des_quat = self.desired_pos[3:]
        robot.des_quat_vel = np.zeros((4,))

        return self.desired_pos

    def getCurrentPos(self, robot):
        return np.hstack((robot.current_c_pos, robot.current_c_quat))
