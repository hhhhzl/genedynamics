import threading
from abc import abstractmethod

import numpy as np

import environments.d3il.d3il_sim.controllers.GainsInterface as gains


class ControllerBase:
    """
    Controller base class.
    """

    def __init__(self):
        self.paramsLock = threading.Lock()
        self.last_control_timestamp = np.NAN
        self._max_duration = None
        self._max_timesteps = None
        self._controller_timer = None

    def isFinished(self, robot):
        """check if controller execution is finished.
        Checks if the Robot Timedata is greater than maxDuration or maxTimesteps.

        Args:
            robot (RobotBase): the robot

        Returns:
            bool: True if execution is finished.
        """
        if self._max_duration is not None:
            return robot.time_stamp - self._controller_timer >= self._max_duration

        if self._max_timesteps is not None:
            return robot.step_count - self._controller_timer >= self._max_timesteps
        return False

    def initController(self, robot, maxDuration):
        return

    def getControl(self, robot):
        self.last_control_timestamp = robot.time_stamp
        return 0

    def is_used(self, robot):
        return (
            not np.isnan(self.last_control_timestamp)
            and robot.time_stamp - self.last_control_timestamp < 0.03
        )

    def setAction(self, action):
        return 0

    def run(self, robot, log=True):
        """Drive the Simulation via the robot.

        Args:
            robot (RobotBase): Robot running the controller
        """
        while not self.isFinished(robot):
            robot.nextStep(log)

    def executeController(self, robot, maxDuration=10, block=True, log=True):
        """Runs the simulation until the position is reached or the maximum duration is exceeded.

        Args:
            robot (RobotBase): Robot running the controller
            maxDuration (int, optional): maximum execution duration. Defaults to 10.
            block (bool, optional): run the simulation now. Defaults to True.
        """
        self._max_duration = maxDuration
        self._max_timesteps = None
        self._controller_timer = robot.time_stamp

        self.initController(robot, maxDuration)
        robot.activeController = self

        if block:
            self.run(robot, log=log)

    def executeControllerTimeSteps(self, robot, timeSteps=10, block=True, log=True):
        """Runs the simulation until the position is reached or the maximum timeSteps is exceeded.

        Args:
            robot (RobotBase): Robot running the controller
            timeSteps (int, optional): maximum number of execution steps. Defaults to 10.
            block (bool, optional): run the simulation now. Defaults to True.
        """
        self._max_duration = None
        self._max_timesteps = timeSteps
        self._controller_timer = robot.step_count

        self.initController(robot, timeSteps * robot.dt)
        robot.activeController = self

        if block:
            self.run(robot, log)

    @abstractmethod
    def reset(self):
        pass


class TorqueController(ControllerBase):
    """
    Controller base class.
    """

    def __init__(self):
        ControllerBase.__init__(self)
        self.reset()

    def getControl(self, robot):
        super(TorqueController, self).getControl(robot)
        return self.torque

    def setAction(self, action):
        self.torque = action.copy()

    def reset(self):
        self.torque = []


class TrackingController(ControllerBase):
    """
    Base class for `JointPDController`, `ZeroTorqueController`. Extends `Controller` class.
    """

    def __init__(self, dimSetPoint):
        ControllerBase.__init__(self)
        self.dimSetPoint = dimSetPoint
        self.tracking_error = False

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
        pass

    def getCurrentPos(self, robot):
        pass

    def getDesiredPos(self, robot):
        pass

    @abstractmethod
    def reset(self):
        pass


class JointPDController(TrackingController, gains.JointPDGains):
    """
    PD Controller for controlling robot joints. Extends `TrackingController` class.
    """

    def __init__(self):
        TrackingController.__init__(self, dimSetPoint=7)
        gains.JointPDGains.__init__(self)

        self.reset()

    def reset(self):
        self.desired_joint_pos = np.array([0, 0, 0, -1.562, 0, 1.914, 0])
        self.desired_joint_vel = np.zeros((7,))
        self.desired_joint_acc = np.zeros((7,))

    def getControl(self, robot):
        """
        Calculates the robot joint acceleration based on
        - the current joint velocity
        - the current joint positions

        :param robot: instance of the robot
        :return: target joint acceleration (num_joints, )
        """
        super(JointPDController, self).getControl(robot)
        self.paramsLock.acquire()
        qd_d = self.desired_joint_pos - robot.current_j_pos
        vd_d = self.desired_joint_vel - robot.current_j_vel

        target_j_acc = self.pgain * qd_d + self.dgain * vd_d  # original

        robot.des_joint_pos = self.desired_joint_pos.copy()
        robot.des_joint_vel = self.desired_joint_vel.copy()
        robot.des_joint_acc = self.desired_joint_acc.copy()

        self.paramsLock.release()
        return target_j_acc

    # setGains.pGain = (21312)

    # @gin.config todo
    # def setGains(self, pGain, dGain):
    #     """
    #     Setter for the gains of the PD Controller.
    #
    #     :param pGain: p gain
    #     :param dGain: d gain
    #     :return: no return value
    #     """
    #     self.paramsLock.acquire()
    #     self.pgain = pGain
    #     self.dgain = dGain
    #     self.paramsLock.release()

    def setSetPoint(self, desired_pos, desired_vel=None, desired_acc=None):
        """
        Sets the desired position, velocity and acceleration of the joints.

        :param desired_pos: desired position (num_joints,)
        :param desired_vel: desired velocity (num_joints,)
        :param desired_acc: desired acceleration (num_joints,)
        :return: no return value
        """
        self.paramsLock.acquire()
        self.desired_joint_pos = desired_pos
        if desired_vel is not None:
            self.desired_joint_vel = desired_vel
        if desired_acc is not None:
            self.desired_joint_acc = desired_acc
        self.paramsLock.release()

    def getCurrentPos(self, robot):
        """
        Getter for the current joint positions.

        :param robot: instance of the robot
        :return: current joint position (num_joints, 1)
        """
        return robot.current_j_pos

    def getDesiredPos(self, robot):
        """
        Getter for the current joint positions.

        :param robot: instance of the robot
        :return: current joint position (num_joints, 1)
        """
        return robot.des_joint_pos


class ModelBasedFeedforwardController(JointPDController):
    """
    PD Controller for controlling robot joints. Extends `TrackingController` class.
    """

    def __init__(self):
        JointPDController.__init__(self)

    def getControl(self, robot):
        """
        Calculates the robot joint acceleration based on
        - the current joint velocity
        - the current joint positions

        :param robot: instance of the robot
        :return: target joint acceleration (num_joints, )
        """
        super(ModelBasedFeedforwardController, self).getControl(robot)
        self.paramsLock.acquire()
        qd_d = self.desired_joint_pos - robot.current_j_pos
        vd_d = self.desired_joint_vel - robot.current_j_vel

        target_j_acc = (
            self.pgain * qd_d + self.dgain * vd_d  # + self.desired_joint_acc
        )  # original
        uff = robot.get_mass_matrix(self.desired_joint_pos).dot(
            self.desired_joint_acc
        ) + robot.get_coriolis(self.desired_joint_pos, self.desired_joint_vel)

        robot.des_joint_pos = self.desired_joint_pos.copy()
        robot.des_joint_vel = self.desired_joint_vel.copy()
        robot.des_joint_acc = self.desired_joint_acc.copy()

        self.paramsLock.release()
        return target_j_acc + uff


class ModelBasedFeedbackController(JointPDController):
    """
    PD Controller for controlling robot joints. Extends `TrackingController` class.
    """

    def __init__(self):
        JointPDController.__init__(self)

    def getControl(self, robot):
        """
        Calculates the robot joint acceleration based on
        - the current joint velocity
        - the current joint positions

        :param robot: instance of the robot
        :return: target joint acceleration (num_joints, )
        """
        super(ModelBasedFeedbackController, self).getControl(robot)
        self.paramsLock.acquire()
        qd_d = self.desired_joint_pos - robot.current_j_pos
        vd_d = self.desired_joint_vel - robot.current_j_vel

        target_j_acc = (
            self.pgain * qd_d + self.dgain * vd_d + self.desired_joint_acc
        )  # original
        uff = robot.get_mass_matrix(robot.current_j_pos).dot(
            target_j_acc
        ) + robot.get_coriolis(robot.current_j_pos, robot.current_j_vel)

        robot.des_joint_pos = self.desired_joint_pos.copy()
        robot.des_joint_vel = self.desired_joint_vel.copy()
        robot.des_joint_acc = self.desired_joint_acc.copy()

        self.paramsLock.release()
        return uff


class JointPositionController(JointPDController):
    def setAction(self, action):
        self.desired_joint_pos = action


class JointVelocityController(JointPDController):
    def __init__(self):
        JointPDController.__init__(self)
        self.pgain = np.zeros((self.dimSetPoint,))

    def setAction(self, action):
        self.desired_joint_vel = action


class ZeroTorqueController(TrackingController):
    """
    Zero torque PD-Controller. Extends `TrackingController` class.
    """

    def __init__(self, dimSetPoint=7):
        TrackingController.__init__(self, dimSetPoint=dimSetPoint)

    def getControl(self, robot):
        super().getControl(robot)
        target_j_acc = np.zeros((self.dimSetPoint,))
        return target_j_acc

    def reset(self):
        pass


class DampingController(ControllerBase, gains.DampingGains):
    """
    Damping (D) Controller.
    """

    def __init__(self):
        ControllerBase.__init__(self)
        gains.DampingGains.__init__(self)

    def getControl(self, robot):
        """
        Calculates the robot joint acceleration based on
        - the current joint velocity

        :param robot: instance of the robot
        :return: target joint acceleration (num_joints, )
        """
        super(DampingController, self).getControl(robot)
        self.paramsLock.acquire()
        target_j_acc = -self.dgain * robot.current_j_vel
        self.paramsLock.release()
        return target_j_acc

    # def setGains(self, dGain):
    #     """
    #     Setter for the gains of the Damping (D) Controller.
    #
    #     :param dGain: d gain
    #     :return: no return value
    #     """
    #
    #     self.paramsLock.acquire()
    #     self.dgain = dGain
    #     self.paramsLock.release()
