from enum import Enum, auto

import numpy as np

import environments.d3il.d3il_sim.controllers as ctrl
import environments.d3il.d3il_sim.core.logger as logger
import environments.d3il.d3il_sim.core.Model as model
import environments.d3il.d3il_sim.core.time_keeper as time_keeper
import environments.d3il.d3il_sim.utils.geometric_transformation as geom_trans


class RobotControlInterface(Enum):
    TorqueInterface = (auto(),)
    CartesianInterface = auto()


class RobotBase:
    """
    This class implements a physics-engine independent robot base class.
    """

    def __init__(self, scene, dt, num_DoF=7, base_position=None, base_orientation=None):
        """
        Init of the robot params.
        """
        self.scene = scene
        self.dt = dt
        self.num_DoF = num_DoF

        self.inhand_cam = None

        if base_position is None:
            base_position = [0.0, 0.0, 0.0]
        self.base_position = np.array(base_position)

        if base_orientation is None:
            base_orientation = [1.0, 0.0, 0.0, 0.0]
        self.base_orientation = np.array(base_orientation)

        self.misc_data = None

        self.use_inv_dyn = False
        self.gravity_comp = True
        self.clip_rate = False

        self.kinModel = model.RobotModelFromPinochio()
        self.dynModel = self.kinModel

        # This attribute is to fix, if we want to use the last planned spline
        # point, for re-planning another spline. If set to true, we will
        # obtain smooth splines.
        self.smooth_spline = True

        self.torque_limit = np.array(
            [80, 80, 80, 80, 10, 10, 10], dtype=np.float64
        )  # Absolute torque limit

        # joint velocity constraints -- more conservative limits from Frankas official website
        self.joint_vel_limit = np.array([2.00, 2.00, 2.00, 2.00, 2.50, 2.50, 2.50])
        self.joint_pos_min = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        )
        self.joint_pos_max = np.array(
            [2.8973, 1.7628, 2.0, -0.0698, 2.8973, 3.7525, 2.8973]
        )

        self.clip_actions = True
        self.rate_limit = 0.8  # Rate torque limit
        self.end_effector = None

        # per time-step controllers
        self.jointTrackingController = ctrl.JointPDController()
        self.cartesianPosTrackingController = ctrl.CartPosImpedenceController()
        self.cartesianPosQuatTrackingController = ctrl.CartPosQuatImpedenceController()
        self.activeController = self.jointTrackingController

        self.jointPosController = ctrl.JointPositionController()
        self.jointVelController = ctrl.JointVelocityController()
        self.torqueController = ctrl.TorqueController()

        self.gotoJointController = ctrl.GotoJointController(self.dt)

        # cartesian space tracking controllers
        self.gotoCartPosController = ctrl.GotoCartPosQuatOfflineIKController(self.dt)
        self.gotoCartPosQuatController = ctrl.GotoCartPosQuatOfflineIKController(
            self.dt
        )

        # using impedance controller
        self.gotoCartPosImpedanceController = ctrl.GotoCartPosImpedanceController(
            self.dt
        )
        self.gotoCartPosQuatImpedanceController = (
            ctrl.GotoCartPosQuatImpedanceController(self.dt)
        )

        # for following a given joint trajectory
        self.jointTrajectoryTracker = ctrl.JointTrajectoryTracker(self.dt)
        self.cartPosQuatTrajectoryTracker = ctrl.CartPosQuatTrajectoryTracker(self.dt)

        self.controlInterface = RobotControlInterface.TorqueInterface

        self.robot_logger = logger.RobotLogger(self)
        self.time_keeper = time_keeper.TimeKeeper(self.dt)
        self.additional_loggers = []

        self.scene.add_robot(self)
        # reset all list or numpy array attributes
        self.reset()

    def reset(self):
        """
        Reset all list or numpy array attributes
        """
        # joints
        self.current_j_pos = np.zeros((self.num_DoF,)) * np.nan
        self.current_j_vel = np.zeros((self.num_DoF,)) * np.nan
        self.des_joint_pos = np.zeros((self.num_DoF,)) * np.nan
        self.des_joint_vel = np.zeros((self.num_DoF,)) * np.nan
        self.des_joint_acc = np.zeros((self.num_DoF,)) * np.nan

        # fingers
        self.current_fing_pos = np.zeros((2,)) * np.nan
        self.current_fing_vel = np.zeros((2,)) * np.nan
        self.des_fing_pos = np.zeros(2) * np.nan
        self.gripper_width = 0
        self.set_gripper_width = 0.001
        self.finger_commands = np.zeros((2,))
        self.grasp_flag = False

        # end effector in local coords
        self.current_c_pos = np.zeros((3,)) * np.nan
        self.current_c_vel = np.zeros((3,)) * np.nan
        self.current_c_quat = np.zeros((4,)) * np.nan
        self.current_c_quat_vel = np.zeros((4,)) * np.nan
        self.des_c_pos = np.zeros((3,)) * np.nan
        self.des_c_vel = np.zeros((3,)) * np.nan
        self.des_quat = np.zeros((4,)) * np.nan
        self.des_quat_vel = np.zeros((4,)) * np.nan

        # end effector in global coords
        self.current_c_pos_global = np.zeros((3,)) * np.nan
        self.current_c_vel_global = np.zeros((3,)) * np.nan
        self.current_c_quat_global = np.zeros((4,)) * np.nan
        self.current_c_quat_vel_global = np.zeros((4,)) * np.nan
        self.des_c_pos_global = np.zeros((3,)) * np.nan
        self.des_c_vel_global = np.zeros((3,)) * np.nan
        self.des_quat_global = np.zeros((4,)) * np.nan
        self.des_quat_vel_global = np.zeros((4,)) * np.nan

        # commands and forces
        # Extend uff to include finger commands for timestep 0
        self.uff = np.zeros((self.num_DoF + 2,))
        self.uff_last = np.zeros((self.num_DoF + 2,))
        self.last_cmd = np.zeros((self.num_DoF,))
        self.command = np.zeros((self.num_DoF,))
        self.grav_terms = np.zeros(9) * np.nan
        self.current_load = np.zeros(
            (self.num_DoF,)
        )  # only for the joints (no fingers) for now

        self.robot_logger = logger.RobotLogger(self)
        self.time_keeper = time_keeper.TimeKeeper(self.dt)

        # torque-based controllers
        if self.gotoJointController is not None:
            self.gotoJointController.resetTrajectory()

        if self.gotoCartPosController is not None:
            self.gotoCartPosController.resetTrajectory()
        if self.gotoCartPosQuatController is not None:
            self.gotoCartPosQuatController.resetTrajectory()
        if self.gotoCartPosImpedanceController is not None:
            self.gotoCartPosImpedanceController.resetTrajectory()
        if self.gotoCartPosQuatImpedanceController is not None:
            self.gotoCartPosQuatImpedanceController.resetTrajectory()

        # reset per time-step controllers

        self.jointTrackingController = ctrl.JointPDController()
        #         self.cartesianPosTrackingController = ctrl.CartPosImpedenceController()
        #         self.cartesianPosQuatTrackingController.reset()
        #         self.activeController = self.jointTrackingController

        # self.jointTrackingController.reset()
        self.cartesianPosTrackingController.reset()
        self.cartesianPosQuatTrackingController.reset()
        self.activeController.reset()

    def getForwardKinematics(self, q=None):
        if q is None:
            q = self.current_j_pos

        return self.kinModel.getForwardKinematics(q)

    def getJacobian(self, q=None):
        if q is None:
            q = self.current_j_pos

        return self.kinModel.getJacobian(q)

    def get_gravity(self, q=None):
        if q is None:
            q = self.current_j_pos

        return self.dynModel.get_gravity(q)

    def get_coriolis(self, q=None, qd=None):
        if q is None:
            q = self.current_j_pos

        if qd is None:
            qd = self.current_j_vel

        return self.dynModel.get_coriolis(q, qd)

    def get_mass_matrix(self, q=None):
        if q is None:
            q = self.current_j_pos
        return self.dynModel.get_mass_matrix(q)

    def start_logging(self, duration: float = 300.0, **kwargs):
        self.robot_logger.start_logging(duration, **kwargs)

    def stop_logging(self):
        self.robot_logger.stop_logging()

    def log_data(self):
        self.robot_logger.log_data()

    def hold_joint_position(self):
        self.des_joint_pos = self.current_j_pos
        self.command = None
        self.activeController = self.jointTrackingController
        self.jointTrackingController.setSetPoint(self.current_j_pos)

    def gotoJointPosition(self, desiredPos, duration=4.0, block=True):
        """
        Moves the joints of the robot in the specified duration to the desired position.
        (in cartesian coordinates).

        :param desiredPos: joint values of the desired position
        :param duration: duration for moving to the position
        :param gains: gains for PD controller
        :return: no return value
        """
        self.gotoJointController.setDesiredPos(desiredPos)
        self.gotoJointController.executeController(self, duration, block=block)

    def executeJointPosCtrlTimeStep(self, action, timeSteps=1, block=True):
        self.jointPosController.setAction(action)
        self.jointPosController.executeControllerTimeSteps(self, timeSteps, block=block)

    def executeJointVelCtrlTimeStep(self, action, timeSteps=1, block=True):
        self.jointVelController.setAction(action)
        self.jointVelController.executeControllerTimeSteps(self, timeSteps, block=block)

    def executeTorqueCtrlTimeStep(self, action, timeSteps=1, block=True):
        # no pd ctrl params needed
        self.torqueController.setAction(action)
        self.torqueController.executeControllerTimeSteps(self, timeSteps, block=block)

    def follow_JointTraj(self, desiredTraj, goto_start=True, block=True):

        if goto_start:
            if np.linalg.norm(self.current_j_pos - desiredTraj[0, :]) > 0.05:
                self.gotoJointPosition(desiredTraj[0, :], 4.0)

        self.jointTrajectoryTracker.setTrajectory(trajectory=desiredTraj)
        self.jointTrajectoryTracker.executeController(
            self, maxDuration=desiredTraj.shape[0] * self.dt, block=block
        )

    def gotoCartPosition_ImpedanceCtrl(
        self, desiredPos, duration=4.0, global_coord=True, block=True
    ):
        if global_coord:
            desiredPos = self._localize_cart_pos(desiredPos)

        self.gotoCartPosImpedanceController.setDesiredPos(desiredPos)
        self.gotoCartPosImpedanceController.executeController(
            self, duration, block=block
        )

    def gotoCartPositionAndQuat_ImpedanceCtrl(
        self, desiredPos, desiredQuat, duration=4.0, global_coord=True, block=True
    ):
        if global_coord:
            desiredPos = self._localize_cart_pos(desiredPos)
            desiredQuat = self._localize_cart_quat(desiredQuat)

        self.gotoCartPosQuatImpedanceController.setDesiredPos(
            np.hstack((desiredPos, desiredQuat))
        )
        self.gotoCartPosQuatImpedanceController.executeController(
            self, duration, block=block
        )

    def gotoCartPosition(self, desiredPos, duration=4.0, global_coord=True, block=True):
        """
        Moves the end effector of the robot in the specified duration to the desired position
        (in cartesian coordinates).

        :param desiredPos: cartesian coordinates of the desired position
        :param duration: duration for moving to the position
        :param global_coord: true if the arguments are in global coordinates. Defaults to True.
        :return: no return value
        """
        if global_coord:
            desiredPos = self._localize_cart_pos(desiredPos)
        self.gotoCartPosController.setDesiredPos(desiredPos)
        self.gotoCartPosController.executeController(self, duration, block=block)

    def gotoCartPositionAndQuat(
        self, desiredPos, desiredQuat, duration=4.0, global_coord=True, block=True, log=True
    ):
        """
        Moves the end effector of the robot in the specified duration to the desired position
        (in cartesian coordinates) with desired orientation (given by quaternion).

        :param desiredPos: cartesian coordinates of the desired position
        :param desiredQuat: orientation given by quaternion
        :param duration: duration for moving to the position
        :param global_coord: true if arguments are in global coordinates. Defaults to True.
        :return:
        """

        if global_coord:
            desiredPos = self._localize_cart_pos(desiredPos)
            desiredQuat = self._localize_cart_quat(desiredQuat)

        self.gotoCartPosQuatController.setDesiredPos(
            np.hstack((desiredPos, desiredQuat))
        )
        self.gotoCartPosQuatController.executeController(self, duration, block=block, log=log)

    # not tested yet
    def follow_CartPositionAndQuatTraj(
        self, desiredPos, desiredQuat, goto_start=True, global_coord=True, block=True
    ):

        if global_coord:
            desiredPos = np.array([self._localize_cart_pos(p) for p in desiredPos])
            desiredQuat = np.array([self._localize_cart_quat(q) for q in desiredQuat])

        desiredTraj = np.hstack((desiredPos, desiredQuat))
        duration = desiredTraj.shape[0] * self.dt

        if goto_start:
            is_position_far = (
                np.linalg.norm(self.current_c_pos - desiredPos[0, :]) > 0.005
            )
            # q1 and q2 are similar if q1 is componentwise similar to q2 or to -q2
            is_rotation_far = (
                np.linalg.norm(self.current_c_quat - desiredQuat[0, :]) > 0.005
            ) and (np.linalg.norm(self.current_c_quat + desiredQuat[0, :]) > 0.005)

            if is_position_far or is_rotation_far:
                self.gotoCartPositionAndQuat(
                    desiredPos=desiredPos[0, :],
                    desiredQuat=desiredQuat[0, :],
                    duration=4.0,
                )

        self.cartPosQuatTrajectoryTracker.setTrajectory(trajectory=desiredTraj)
        self.cartPosQuatTrajectoryTracker.executeController(
            self, maxDuration=duration, block=block
        )

    """
    def executeCartPositionAndQuatImpedanceCtrlTimeStep(self, action, timeSteps=1, gains=None):
        data = self.config.load_yaml('cartAndOr_ctrl_gains')
        data_joint_pd = self.config.load_yaml('PD_control_gains')
        pgain = np.array(data['cartAndOr_ctrl_pgain'], dtype=np.float64)
        dgain = np.array(data['cartAndOr_ctrl_dgain'], dtype=np.float64)
        pgain_null = np.array(data['cartAndOr_ctrl_pgain_null'],
                              dtype=np.float64)
        dgain_null = np.array(data['cartAndOr_ctrl_dgain_null'],
                              dtype=np.float64)
        dgain_joint_pd = np.array(data_joint_pd['dgain'], dtype=np.float64)
        # if self isinstance(Py)
        self.CartPosQuatControllerImpedance.setGains(pGain=pgain, dGain=dgain,
                                                                            pGain_null=pgain_null,
                                                                            dGain_null=dgain_null,
                                                                            dGainVelCtrl=dgain_joint_pd)
        self.CartPosQuatControllerImpedance.setDesiredPos(
            np.hstack((desiredPos, desiredQuat)))
        self.CartPosQuatControllerImpedance.executeController(self, duration)
    """

    def _localize_cart_pos(self, pos):
        # Check if the transformation would do anything at all
        if not np.any(self.base_position) and np.array_equal(
            self.base_orientation, [1, 0, 0, 0]
        ):
            return pos

        pos_vec = np.array(pos)

        local_pos = pos_vec - self.base_position
        local_pos = geom_trans.quat_rot_vec(
            geom_trans.quat_conjugate(self.base_orientation), local_pos
        )
        return local_pos

    def _localize_cart_quat(self, quat):
        # Check if the transformation would do anything at all
        if np.array_equal(self.base_orientation, [1, 0, 0, 0]):
            return quat

        local_quat = quat
        if local_quat is not None:
            quat_vec = np.array(local_quat)
            local_quat = geom_trans.quat_mul(
                geom_trans.quat_conjugate(self.base_orientation), quat_vec
            )
        return local_quat

    def _localize_cart_coords(self, pos, quat=None):
        if quat is not None:
            quat = self._localize_cart_quat(quat)
        return self._localize_cart_pos(pos), quat

    def wait(self, duration=1.0, block=True):
        self.activeController.executeController(self, duration, block=block)

    def open_fingers(self):
        self.set_desired_gripper_width(0.04)

    def close_fingers(self, duration=0.5):
        self.set_gripper_width = 0.0
        self.grasp_flag = False
        if duration > 0:
            self.wait(duration)
        self.grasp_flag = True

    def set_desired_gripper_width(self, desired_width):
        self.set_gripper_width = desired_width
        self.grasp_flag = False

    def fing_ctrl_step(self):
        """
        Calculates the control for the robot finger joints.

        :return: controlling for the robot finger joints with dimension: (num_finger_joints, )
        """
        # pgain = 1 * np.array([200, 200], dtype=np.float64)
        pgain = 1 * np.array([500, 500], dtype=np.float64)
        # dgain = np.sqrt(pgain) #1 * np.array([10, 10], dtype=np.float64)
        dgain = 1 * np.array(
            [10, 10], dtype=np.float64
        )  # 1 * np.array([10, 10], dtype=np.float64)

        gripper_width = self.current_fing_pos
        gripper_width_vel = self.current_fing_vel

        des_fing_pos = np.array(
            [self.set_gripper_width, self.set_gripper_width], dtype=np.float64
        )
        # enforce equal distance of both fingers

        mean_finger_pos = np.mean(gripper_width)
        force = pgain * (mean_finger_pos - gripper_width)

        if (np.mean(gripper_width) - self.set_gripper_width) > 0.005:
            if self.grasp_flag:
                force2 = np.array([-20, -20])
            else:
                force2 = dgain * (np.array([-0.2, -0.2]) - gripper_width_vel)
        else:
            force2 = pgain * (des_fing_pos - gripper_width) - dgain * gripper_width_vel
            force2 = np.clip(force2, -5, 5)

        force = force + force2

        return force

    def receiveState(self):
        """
        Receives the current robot state from the simulation i.e.
        - joint positions, joint velocities by calling `get_qdq_joints_fingers`
        - cartesian coords, velocity and and orientation of the end-effector by calling `get_x()`
        - joint forces by calling `get_joint_reaction_forces`
        - gripper width by calling `getJointState` with finger joint IDs

        --------------------------------------------------------------------------------------------------------------
        Note: PyBullet's quaternion information is always given as [x, y, z, w]
              SL uses the notation: [w, x, y, z]
              We therefore have to do a quick reorganization
              Note that the orientation is also logged in [w, x, y, z]
        --------------------------------------------------------------------------------------------------------------

        :return: no return value
        """
        raise NotImplementedError

    def tick(self):
        self.time_keeper.tick()

    def nextStep(self, log=True):
        """legacy method used by the controllers to run the simulation.
        The function call is now 'redirected' to the scene to support Multibots.
        """
        self.scene.next_step(log)

    def prepare_step(self):
        """
        Executes the simulation for one time step, i.e. calling PyBullet's stepSimulation() which will perform all
        the actions in a single forward dynamics simulation step such as collision detection, constraint solving and
        integration.

        :return: no return value
        """
        self.des_joint_pos[:] = np.nan
        self.des_joint_vel[:] = np.nan
        self.des_joint_acc[:] = np.nan

        self.des_c_pos[:] = np.nan
        self.des_c_vel[:] = np.nan
        self.des_quat[:] = np.nan
        self.des_quat_vel[:] = np.nan

    def get_command_from_inverse_dynamics(
        self, target_j_acc, mj_calc_inv=False, robot_id=None, client_id=None
    ):
        # Do not forget to do a distuinction for mujoco..... if we only do want to have gravity comp,
        # then we will not call the inverse dynamics function.... only sim.data.qfrc_bias !!!
        raise NotImplementedError

    def preprocessCommand(self, target_j_acc):
        if len(target_j_acc.shape) == 2:
            target_j_acc = target_j_acc.reshape((-1))
        if target_j_acc.shape[0] != self.num_DoF:
            raise ValueError(
                "Specified motor command vector needs to be of size {} is {}\n".format(
                    self.num_DoF, target_j_acc.shape[0]
                )
            )

        self.finger_commands = self.fing_ctrl_step()
        if self.use_inv_dyn:
            target_j_acc = np.append(target_j_acc, self.finger_commands[0])
            target_j_acc = np.append(target_j_acc, self.finger_commands[1])
            self.des_joint_acc = target_j_acc.copy()
            self.uff = self.get_command_from_inverse_dynamics(
                target_j_acc, mj_calc_inv=True
            )
            self.uff[7:9] = self.finger_commands

        else:
            target_j_acc = np.append(target_j_acc, self.finger_commands[0])
            target_j_acc = np.append(target_j_acc, self.finger_commands[1])
            self.uff = target_j_acc
            if self.gravity_comp:
                comp_forces = self.get_command_from_inverse_dynamics(
                    target_j_acc=np.zeros(9), mj_calc_inv=False
                )
                self.grav_terms = comp_forces.copy()
                self.uff += comp_forces

        # Catch start of control
        if self.step_count == 0:
            self.uff_last = 0

        if self.clip_rate:
            # Clip and rate limit torque
            uff_diff = self.uff - self.uff_last
            uff_diff = np.clip(uff_diff, -self.rate_limit, self.rate_limit)
            self.uff = self.uff_last + uff_diff

        if self.clip_actions:
            self.uff[:7] = np.clip(self.uff[:7], -self.torque_limit, self.torque_limit)

    def beam_to_cart_pos_and_quat(self, desiredPos, desiredQuat):
        q = self.gotoCartPosQuatController.trajectory_generator.findJointPosition(
            self, desiredPos=desiredPos, desiredQuat=desiredQuat
        )
        self.beam_to_joint_pos(q)

    def beam_to_joint_pos(self, desiredJoints, run=True, log=False):
        self.set_q(desiredJoints)
        self.receiveState()
        self.gotoJointController.resetTrajectory()
        self.gotoCartPosQuatController.resetTrajectory()
        self.gotoCartPosImpedanceController.resetTrajectory()
        self.jointTrackingController.setSetPoint(desiredJoints)
        # self.cartesianPosQuatTrackingController = ctrl.CartPosQuatImpedenceController()
        if run:
            self.jointTrackingController.executeControllerTimeSteps(self, 1, log)

    def set_q(self, joints, robot_id=None, physicsClientId=None):
        """
        Sets the value of the robot joints.
        WARNING: This overrides the physics, do not use during simulation!!

        :param joints: tuple of size (7)
        :return: no return value
        """
        raise NotImplementedError

    @property
    def step_count(self):
        return self.time_keeper.step_count

    @property
    def time_stamp(self):
        return self.time_keeper.time_stamp

    @property
    def step_count(self):
        return self.time_keeper.step_count