"""
This module uses a controller and a desired position to calculate the trajectory of the robot joints.
"""
import logging

import numpy as np
from scipy.interpolate import make_interp_spline

import environments.d3il.d3il_sim.utils as utils
from environments.d3il.d3il_sim.controllers import ControllerBase


class TrajectoryGenerator:
    def __init__(self):
        pass

    def generate_trajectory(self, trackingController, robot, duration):
        pass

    def setDesiredPos(self, desiredPos):
        pass


class TrajectoryTracker(ControllerBase):
    """
    Base class for controller tracking trajectories. Extends the controller base class.
    """

    def __init__(self, dt, trajectory_generator=None):
        ControllerBase.__init__(self)

        self.trajectory_generator = trajectory_generator
        self.startingTime = None
        self.trajectory = None
        self.trajectoryVel = None
        self.trajectoryAcc = None
        self.dt = dt
        self.additionalDuration = 0
        self.delay_trajectory = 0
        self.old_time_stamp = None
        self.delay_trajectory_on_error = False

    def initController(self, robot, maxDuration):
        """
        Initialization of the controller.

        :param robot: instance of the robot
        :param maxDuration: maximal control duration
        :return: no return value
        """
        # robot.receiveState()
        self.get_tracking_controller(robot).initController(robot, maxDuration)
        self.startingTime = robot.time_stamp  # Current robot time stamp
        self.duration = maxDuration
        self.delay_trajectory = 0
        self.old_time_stamp = robot.time_stamp
        if self.trajectory_generator is not None:
            self.setTrajectory(
                self.trajectory_generator.generate_trajectory(
                    self.get_tracking_controller(robot), robot, maxDuration
                )
            )

        self.set_point_for_time_step(robot, 0)

    def get_tracking_controller(self, robot):
        pass

    def setAction(self, action):
        # Hacky Switch between GoToPos and TrajectoryTracking Behavior.
        # Used by MetaRobot f. MultiRobot Commands
        if action.ndim == 1:
            self.setDesiredPos(action)
        else:
            self.setTrajectory(action)

    def setDesiredPos(self, desiredPosition):
        self.trajectory_generator.setDesiredPos(desiredPosition)

    def set_point_for_time_step(self, robot, timeStep):
        desired_pos = self.trajectory[timeStep, :]

        if timeStep < self.trajectory.shape[0] - 1:
            desired_vel = self.trajectoryVel[timeStep, :]
        else:
            desired_vel = np.zeros((self.trajectory.shape[1],))

        if timeStep < self.trajectory.shape[0] - 2:
            desired_acc = self.trajectoryAcc[timeStep, :]
        else:
            desired_acc = np.zeros((self.trajectory.shape[1],))

        self.get_tracking_controller(robot).setSetPoint(
            desired_pos, desired_vel, desired_acc
        )

    def getControl(self, robot):
        if self.trajectory is None:
            logging.getLogger(__name__).warning("Error: Trajectory is empty")

        self.paramsLock.acquire()

        # if tracking is too slow then delay trajectory to allow robot to catch up
        if (
            self.delay_trajectory_on_error
            and self.get_tracking_controller(robot).tracking_error
        ):
            self.delay_trajectory += robot.time_stamp - self.old_time_stamp
        self.old_time_stamp = robot.time_stamp

        timeStep = np.round(
            (robot.time_stamp - self.startingTime - self.delay_trajectory) / self.dt
        )

        timeStep = int(np.min([timeStep, self.trajectory.shape[0] - 1]))

        self.set_point_for_time_step(robot, timeStep)

        self.paramsLock.release()

        return self.get_tracking_controller(robot).getControl(robot)

    def setTrajectory(self, trajectory):
        """
        Set the trajectory from splines.
        :param trajectory: numpy array (num_time_stamps, num_joints)
        :return: no return value
        """
        self.paramsLock.acquire()

        self.trajectory = trajectory
        self.trajectoryVel = np.diff(trajectory, 1, axis=0) / self.dt
        self.trajectoryAcc = np.diff(trajectory, 2, axis=0) / (self.dt**2)

        self.paramsLock.release()

    def resetTrajectory(self):
        """
        Sets the trajectory object to None (used if we expect discontinuities)
        """
        self.trajectory = None


class JointTrajectoryTracker(TrajectoryTracker):
    """
    Tracker for trajectory of the robot joints.
    """

    def __init__(self, dt, trajectory_generator=None):
        TrajectoryTracker.__init__(self, dt, trajectory_generator)

    def get_tracking_controller(self, robot):
        return robot.jointTrackingController


class CartPosTrajectoryTracker(TrajectoryTracker):
    """
    Tracker for the cartesian coordinates of the robot end effector.
    """

    def __init__(self, dt, trajectory_generator=None):
        TrajectoryTracker.__init__(self, dt, trajectory_generator)

    def get_tracking_controller(self, robot):
        return robot.cartesianPosTrackingController


class CartPosQuatTrajectoryTracker(TrajectoryTracker):
    """
    Tracker for the cartesian coordinates and orientation using quaternions of the robot end effector.
    """

    def __init__(self, dt, trajectory_generator=None):
        TrajectoryTracker.__init__(self, dt, trajectory_generator)

    def get_tracking_controller(self, robot):
        return robot.cartesianPosQuatTrackingController


class GotoTrajectoryBase(TrajectoryGenerator):
    def __init__(self):
        """
        Initializes the tracker for the robots trajectory and sets the default value for the duration and
        joint positions.

        :param tracker: tracks robot trajectory
        """

        TrajectoryGenerator.__init__(self)

        self.desiredPosition = np.array(
            [0, 0, 0, -1.562, 0, 1.914, 0]
        )  # default joint positions

    def get_init_pos(self, trackingController, robot):
        # robot.receiveState()

        cur_state = trackingController.getCurrentPos(robot)
        des_state = trackingController.getDesiredPos(robot)
        if (
            robot.smooth_spline
            and not any(np.isnan(des_state))
            and not trackingController.tracking_error
            and trackingController.is_used(robot)
        ):
            cur_state = des_state
        return cur_state

    def setDesiredPos(self, desiredPosition):
        """
        Sets the desired positions of the robot joints.

        :param desiredPosition: numpy array with dim [num_joints,]
        :return: no return value
        """
        self.desiredPosition = desiredPosition


class GotoTrajectorySpline(GotoTrajectoryBase):
    """
    This class sets the robot trajectory with :func:`initController`. The end effector position is set with
    :func:`setDesiredPos`.
    """

    def __init__(self):
        """
        Initializes the tracker for the robots trajectory and sets the default value for the duration and
        joint positions.

        :param tracker: tracks robot trajectory
        """

        GotoTrajectoryBase.__init__(self)

    def generate_trajectory(self, trackingController, robot, duration):

        super().generate_trajectory(trackingController, robot, duration)

        cur_state = self.get_init_pos(trackingController, robot)

        time = np.linspace(
            0, duration, int(duration / robot.dt) + 1
        )  # create time stamp array
        trajectory = np.zeros(
            (time.shape[0], trackingController.dimSetPoint)
        )  # create empty trajectory array

        for i in range(trackingController.dimSetPoint):
            try:
                # This creates a b spline with 0 1st and 2nd order derivatives at the boundaries
                l, r = [(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)]
                bsplinef = make_interp_spline(
                    x=[0, duration],
                    y=[cur_state[i], self.desiredPosition[i]],
                    bc_type=(l, r),
                    k=5,
                )
                trajectory[:, i] = bsplinef(time)
            except ValueError:
                raise ValueError(
                    "Robot might be already be at this configuration.\nDesired Pos: {}\nCurrent Pos: {}".format(
                        self.desiredPosition[i], cur_state[i]
                    )
                )
        return trajectory


class GotoTrajectoryLinear(GotoTrajectoryBase):
    """
    This class sets the robot trajectory with :func:`initController`. The end effector position is set with
    :func:`setDesiredPos`.
    """

    def __init__(self):
        """
        Initializes the tracker for the robots trajectory and sets the default value for the duration and
        joint positions.

        :param tracker: tracks robot trajectory
        """

        GotoTrajectoryBase.__init__(self)

    def generate_trajectory(self, trackingController, robot, duration):

        super().generate_trajectory(trackingController, robot, duration)

        cur_state = self.get_init_pos(trackingController, robot)

        time = np.linspace(
            0, duration, int(duration / robot.dt) + 1
        )  # create time stamp array
        trajectory = np.zeros(
            (time.shape[0], trackingController.dimSetPoint)
        )  # create empty trajectory array

        for i in range(trackingController.dimSetPoint):
            trajectory[:, i] = np.linspace(
                cur_state[i], self.desiredPosition[i], int(duration / robot.dt) + 1
            )

        trajectory = trajectory[1:, :]
        return trajectory


class OfflineIKTrajectoryGenerator(TrajectoryGenerator):
    def __init__(self, interpolation_trajectory=None):
        TrajectoryGenerator.__init__(self)

        self.desiredTaskPosition = np.zeros(
            7,
        )
        self.J_reg = 1e-6  # Jacobian regularization constant
        self.W = np.diag([1, 1, 1, 1, 1, 1, 1])
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
        if interpolation_trajectory is None:
            self.interpolation_trajectory = GotoTrajectorySpline()
        else:
            self.interpolation_trajectory = interpolation_trajectory

    def generate_trajectory(self, trackingController, robot, duration):

        eps = 1e-5
        IT_MAX = 1000
        DT = 1e-3

        i = 0
        self.pgain = [
            33.9403713446798,
            30.9403713446798,
            33.9403713446798,
            27.69370238555632,
            33.98706171459314,
            30.9185531893281,
        ]
        self.pgain_null = 5 * np.array(
            [
                7.675519770796831,
                2.676935478437176,
                8.539040163444975,
                1.270446361314313,
                8.87752182480855,
                2.186782233762969,
                4.414432577659688,
            ]
        )
        self.pgain_limit = 20

        q = robot.current_j_pos.copy()
        qd_d = np.zeros(q.shape)
        oldErrNorm = np.inf
        while True:
            oldQ = q
            q = q + DT * qd_d

            q = np.clip(q, robot.joint_pos_min, robot.joint_pos_max)

            cartPos, orient = robot.getForwardKinematics(q)
            cpos_err = self.desiredTaskPosition[:3] - cartPos

            if np.linalg.norm(orient - self.desiredTaskPosition[3:]) > np.linalg.norm(
                orient + self.desiredTaskPosition[3:]
            ):
                orient = -orient

            cpos_err = np.clip(cpos_err, -0.1, 0.1)
            cquat_err = np.clip(
                utils.get_quaternion_error(orient, self.desiredTaskPosition[3:]),
                -0.5,
                0.5,
            )
            err = np.hstack((cpos_err, cquat_err))

            errNorm = np.sum(cpos_err**2) + np.sum(
                (orient - self.desiredTaskPosition[3:]) ** 2
            )
            if errNorm > oldErrNorm:
                q = oldQ
                DT = DT * 0.7
                continue
            else:
                DT = DT * 1.025

            if errNorm < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break

            # if not i % 1:
            #    logging.getLogger(__name__).debug('%d: error = %s, %s, %s' % (i, errNorm, oldErrNorm, DT))

            oldErrNorm = errNorm

            J = robot.getJacobian(q)

            Jw = J.dot(self.W)

            # J *  W * J' + reg * I
            JwJ_reg = Jw.dot(J.T) + self.J_reg * np.eye(J.shape[0])

            # Null space movement
            qd_null = self.pgain_null * (self.target_th_null - q)

            margin_to_limit = 0.1
            qd_null_limit = np.zeros(qd_null.shape)
            qd_null_limit_max = self.pgain_limit * (
                robot.joint_pos_max - margin_to_limit - q
            )
            qd_null_limit_min = self.pgain_limit * (
                robot.joint_pos_min + margin_to_limit - q
            )
            qd_null_limit[
                q > robot.joint_pos_max - margin_to_limit
            ] += qd_null_limit_max[q > robot.joint_pos_max - margin_to_limit]
            qd_null_limit[
                q < robot.joint_pos_min + margin_to_limit
            ] += qd_null_limit_min[q < robot.joint_pos_min + margin_to_limit]

            qd_null += qd_null_limit
            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, self.pgain * err - J.dot(qd_null))
            # qd_d = self.pgain * err
            qd_d = self.W.dot(J.transpose()).dot(qd_d) + qd_null

            i += 1

        print("Final IK error (%d iterations):  %s" % (i, errNorm))

        logging.getLogger(__name__).debug(
            "Final IK error (%d iterations):  %s" % (i, errNorm)
        )
        self.interpolation_trajectory.setDesiredPos(q)
        return self.interpolation_trajectory.generate_trajectory(
            trackingController, robot, duration
        )

    def setDesiredPos(self, desiredTaskPosition):
        """
        Sets the desired positions of the robot joints.

        :param desiredPosition: numpy array with dim [num_joints,]
        :return: no return value
        """
        self.desiredTaskPosition = desiredTaskPosition


class GotoCartPosQuatOfflineIKController(JointTrajectoryTracker):
    def __init__(self, dt):
        super(JointTrajectoryTracker, self).__init__(
            dt, trajectory_generator=OfflineIKTrajectoryGenerator()
        )


class GotoJointController(JointTrajectoryTracker):
    def __init__(self, dt):
        super(JointTrajectoryTracker, self).__init__(
            dt, trajectory_generator=GotoTrajectorySpline()
        )


class GotoCartPosImpedanceController(CartPosTrajectoryTracker):
    def __init__(self, dt):
        super(CartPosTrajectoryTracker, self).__init__(
            dt, trajectory_generator=GotoTrajectorySpline()
        )


class GotoCartPosQuatImpedanceController(CartPosQuatTrajectoryTracker):
    def __init__(self, dt):
        super(CartPosQuatTrajectoryTracker, self).__init__(
            dt, trajectory_generator=GotoTrajectorySpline()
        )


class GotoCartPosCartesianRobotController(JointTrajectoryTracker):
    """
    Controller for the cartesian coordinates of the robot.
    """

    def __init__(self, dt):
        assert False  # not implemented yet
        super(JointTrajectoryTracker, self).__init__(
            dt, trajectory_generator=GotoTrajectory()
        )


class GotoCartPosQuatCartesianRobotController(JointTrajectoryTracker):
    """
    Controller for the cartesian coordinates and the orientation (using quaternions) of the robot.
    """

    def __init__(self, dt):
        assert False  # not implemented yet
        super(JointTrajectoryTracker, self).__init__(
            dt, trajectory_generator=GotoTrajectory()
        )
