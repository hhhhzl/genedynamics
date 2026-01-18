import logging

import numpy as np
import pybullet

from environments.d3il.d3il_sim.core import RobotBase
from environments.d3il.d3il_sim.sims.pybullet.PybulletCamera import PbInHandCamera
from environments.d3il.d3il_sim.utils.geometric_transformation import wxyz_to_xyzw
from environments.d3il.d3il_sim.utils.sim_path import d3il_path


class PyBulletRobot(RobotBase):
    def __init__(
        self,
        scene,
        dt=1e-3,
        num_DoF=7,
        base_position=None,
        base_orientation=None,
        gravity_comp=True,
        clip_actions=False,
        pos_ctrl=False,
        path_to_urdf: str = None,
    ):

        super(PyBulletRobot, self).__init__(
            scene=scene,
            dt=dt,
            num_DoF=num_DoF,
            base_position=base_position,
            base_orientation=base_orientation,
        )

        self.gravity_comp = gravity_comp
        self.clip_actions = clip_actions

        self.pos_ctrl = pos_ctrl

        self.client_id = None
        self.robot_id = None
        self.robot_id_ik = None
        self.robotEndEffectorIndex = None
        self.physics_client_id = None

        if path_to_urdf is None:
            path_to_urdf = d3il_path(
                "./models/pybullet/robots/panda_arm_hand.urdf"
            )
        self.path_to_urdf = path_to_urdf

        # Add the inhand camera here already to allow changing of parameters before start
        self.inhand_cam = PbInHandCamera()

    def get_qdq_J(self, robot_id=None, client_id=None):
        """
        This method calculates the joint positions, the joint velocities and the Jacobian.
        Note the position and the velocity of the fingers is not included here.

        :return: q: joint positions
                dq: joint velocities
                 J: jacobian matrix (6x7)
        """
        if client_id is None:
            client_id = self.physics_client_id
        if robot_id is None:
            robot_id = self.robot_id

        qdq_matrix = np.array(
            [
                np.array(
                    pybullet.getJointState(
                        bodyUniqueId=robot_id,
                        jointIndex=jointIndex,
                        physicsClientId=client_id,
                    )[:2]
                )
                for jointIndex in np.arange(1, 8)
            ]
        )
        q = qdq_matrix[:, 0]
        dq = qdq_matrix[:, 1]

        jac_t, jac_r = pybullet.calculateJacobian(
            robot_id,
            self.robotEndEffectorIndex,
            [0.0, 0.0, 0.0],
            list(q) + [0.0] * 2,
            [0.0] * 9,
            [0.0] * 9,
            physicsClientId=client_id,
        )

        J = np.concatenate((np.array(jac_t)[:, :7], np.array(jac_r)[:, :7]), axis=0)
        return np.array(q), np.array(dq), J

    def get_qdq_fingers(self, robot_id=None, client_id=None):
        """
        This method returns the position and the velocities of the fingers.

        :return: fing_pos: 2x1 position of both fingers as np array
                 fing_vel: 2x1 velocity of both fingers as np array
        """
        if robot_id is None:
            robot_id = self.robot_id
        if client_id is None:
            client_id = self.physics_client_id
        f1_info = pybullet.getJointState(
            bodyUniqueId=robot_id, jointIndex=10, physicsClientId=client_id
        )
        f2_info = pybullet.getJointState(
            bodyUniqueId=robot_id, jointIndex=11, physicsClientId=client_id
        )

        fing_pos = np.array([f1_info[0], f2_info[0]])
        fing_vel = np.array([f1_info[1], f2_info[1]])
        return fing_pos, fing_vel

    def get_qdq_joints_fingers(self, robot_id=None, client_id=None):
        """
        This method returns position and velocity of the joints and the fingers combined in one array.

        :return: joint and finger positions as np array (9x1)
                 joint and finger velocities as np array (9x1)
        """
        if robot_id is None:
            robot_id = self.robot_id
        if client_id is None:
            client_id = self.physics_client_id
        qdq_matrix = np.array(
            [
                np.array(
                    pybullet.getJointState(
                        bodyUniqueId=robot_id,
                        jointIndex=jointIndex,
                        physicsClientId=client_id,
                    )[:2]
                )
                for jointIndex in np.arange(1, 8)
            ]
        )
        q = qdq_matrix[:, 0]
        dq = qdq_matrix[:, 1]
        q = list(q)
        dq = list(dq)
        q_dq_finger = self.get_qdq_fingers()
        q.append(q_dq_finger[0][0])
        q.append(q_dq_finger[0][1])
        dq.append(q_dq_finger[1][0])
        dq.append(q_dq_finger[1][1])
        return np.array(q), np.array(dq)

    def get_joint_reaction_forces(self, robot_id=None, client_id=None):
        """
        Callback to PyBullets `getJointState` to calculate the joint reaction forces.

        :param robot_id: robot ID returned by calling `loadURDF`
        :param client_id: ID of the physics client
        :return: joint reaction forces (num joints, ) with ||Fx, Fy, Fz|| for each joint
        """
        if robot_id is None:
            robot_id = self.robot_id
        if client_id is None:
            client_id = self.physics_client_id
        forces = np.zeros(self.num_DoF)
        for joint in self.jointIndices:
            infs = pybullet.getJointState(
                bodyUniqueId=robot_id, jointIndex=joint, physicsClientId=client_id
            )[2]
            forces[joint - 1] = np.linalg.norm(np.array(infs[0:3]))
        return forces

    def get_x(self, robot_id=None, client_id=None):
        """
        This method returns the cartesian world position, the cartesian velocity and the quaternion
        orientation of the end effector by calling pyBullets `getLinkState`

        :return: robot_x: cartesian world coordinates of end effector
                 robot_dx_dt: cartesian velocity of end effector
                 robot_quat: quaternion end effector orientation
        """
        if robot_id is None:
            robot_id = self.robot_id
        if client_id is None:
            client_id = self.physics_client_id

        # link_infos[0]: Cartesian position of center of mass
        # link_infos[1]: Cartesian orientation of center of mass, in quaternion [x,y,z,w]
        # link_infos[2]: local position offset of inertial frame (center of mass) expressed in the URDF link frame
        # link_infos[3]: local orientation offset of the inertial frame expressed in URDF link frame.
        # link_infos[4]: world position of the URDF link frame
        # link_infos[5]: world orientation of the URDF link frame
        # link_infos[6]: Cartesian world velocity. Only returned if computeLinkVelocity non-zero.

        link_infos = pybullet.getLinkState(
            robot_id,
            linkIndex=self.robotEndEffectorIndex,
            computeLinkVelocity=1,  # Cartesian world velocity will be returned
            physicsClientId=client_id,
        )
        robot_x = np.array(link_infos[4])
        robot_dx_dt = np.array(link_infos[6])
        robot_quat = np.array(link_infos[5])  # quaternion: [x,y,z,w]
        return robot_x, robot_dx_dt, robot_quat

    def get_command_from_inverse_dynamics(
        self, target_j_acc, mj_calc_inv=False, robot_id=None, client_id=None
    ):
        """
        This method uses the calculation of the inverse Dynamics method of pybullet. Note, that all parameters have to
        be in list format. Otherwise and segmentation get_error is returned.
        Notes on the calculateInverseDynamics function:
            The calculateInverseDynamics function NEEDS all degrees of freedom, which includes also the fingers, since
            they are marked as prismatic joints in the URDF. Only 'fixed joints and the base joint' can be skipped and
            need not to be included in the position and the velocity vectors

        :param q: joint positions and finger positions (have to be given)
        :param client_id: client id of the simulation
        :param robot_id: robot id in the scene
        :param dq: joint velocities and finger velocities
        :param desired_acceleration: The desired acceleration of each degree of freedom as list
        :return: torques for each degree of freedom (9) to achieve the desired acceleration
        """
        if robot_id is None:
            robot_id = self.robot_id
        if client_id is None:
            client_id = self.physics_client_id
        q, dq = self.get_qdq_joints_fingers()
        torques = pybullet.calculateInverseDynamics(
            bodyUniqueId=robot_id,
            objPositions=list(q),
            objVelocities=list(dq),
            objAccelerations=list(target_j_acc),
            physicsClientId=client_id,
        )
        return np.array(torques)

    def get_invKinematics(
        self, targetPosition, targetOrientation, robot_id=None, client_id=None
    ):
        if robot_id is None:
            robot_id = self.robot_id
        if client_id is None:
            client_id = self.physics_client_id
        des_orientation = np.zeros(4)
        des_orientation[:3] = targetOrientation[1::]
        des_orientation[-1] = targetOrientation[0]
        des_joints = pybullet.calculateInverseKinematics(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=self.robotEndEffectorIndex,
            targetPosition=list(targetPosition),
            targetOrientation=list(des_orientation),
            residualThreshold=1e-4,
            maxNumIterations=4000,
            physicsClientId=client_id,
        )
        return np.array(des_joints)[:7]

    def receiveState(self):
        """
        Receives the current state i.e.
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

        states = self.get_qdq_joints_fingers()
        current_j_pos = states[0][:7]  # joint positions
        current_j_vel = states[1][:7]  # joint velocities
        self.current_j_vel = current_j_vel
        self.current_j_pos = current_j_pos

        self.current_fing_pos = states[0][7:9]  # finger positions
        self.current_fing_vel = states[1][7:9]  # finger velocities

        cart_infos = self.get_x()
        self.current_c_pos_global = cart_infos[0]
        self.current_c_vel_global = cart_infos[1]

        c_quat = np.zeros(4)
        c_quat[1::] = cart_infos[2][:3]
        c_quat[0] = cart_infos[2][-1]
        self.current_c_quat_global = c_quat  # [w , x, y, z]
        self.current_c_quat_vel_global = np.zeros(4)
        self.current_c_quat_vel_global[1:] = self.current_c_vel_global
        self.current_c_quat_vel_global *= 0.5 * self.current_c_quat_global

        ### ENDEFFECTOR LOCAL
        self.current_c_pos, self.current_c_quat = self._localize_cart_coords(
            self.current_c_pos_global, self.current_c_quat_global
        )
        self.current_c_vel, _ = self._localize_cart_coords(
            # add base_position, as it is subtracted in _localize_cart_coords
            self.current_c_vel_global
            + self.base_position
        )
        # This must be checked!
        _, self.current_c_quat_vel = self._localize_cart_coords(
            self.base_position, self.current_c_quat_vel_global
        )

        self.current_load = self.get_joint_reaction_forces()

        # calculate width of the fingers:
        # I don't know if this is correct! Check later!
        self.gripper_width = np.abs(
            pybullet.getJointState(bodyUniqueId=self.robot_id, jointIndex=10)[0]
            - pybullet.getJointState(bodyUniqueId=self.robot_id, jointIndex=11)[0]
        )
        self.last_cmd = self.uff

    def prepare_step(self):
        """
        Executes the simulation for one time stamp, i.e. calling PyBullet's stepSimulation() which will perform all
        the actions in a single forward dynamics simulation step such as collision detection, constraint solving and
        integration.

        :return: no return value
        """
        self.command = self.activeController.getControl(self)
        self.preprocessCommand(self.command)
        # set the torques in pybullet
        if self.pos_ctrl:
            max_forces = self.torque_limit
            posGains = np.ones(7) * 0.0015
            posGains[3] *= 10
            posGains[5] *= 10

            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.jointIndices_with_fingers[:-2],
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=None,
            )
            finger_commands = self.fing_ctrl_step()
            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=[10, 11],
                controlMode=pybullet.TORQUE_CONTROL,
                forces=finger_commands,
            )
        else:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.jointIndices_with_fingers,
                controlMode=pybullet.TORQUE_CONTROL,
                forces=self.uff.copy(),
            )

    def beam_to_cart_pos_and_quat(self, desiredPos, desiredQuat):
        des_joints = self.get_invKinematics(desiredPos, desiredQuat)
        self.beam_to_joint_pos(des_joints)

    def set_q(self, joints, robot_id=None, physicsClientId=None):
        """
        Sets the value of the robot joints.
        WARNING: This overrides the physics, do not use during simulation!!

        :param joints: tuple of size (7)
        :return: no return value
        """
        if physicsClientId is None:
            physicsClientId = self.physics_client_id
        if robot_id is None:
            robot_id = self.robot_id
        j1, j2, j3, j4, j5, j6, j7 = joints

        joint_angles = {}
        joint_angles["panda_joint_world"] = 0.0  # No actuation
        joint_angles["panda_joint1"] = j1
        joint_angles["panda_joint2"] = j2
        joint_angles["panda_joint3"] = j3
        joint_angles["panda_joint4"] = j4
        joint_angles["panda_joint5"] = j5
        joint_angles["panda_joint6"] = j6
        joint_angles["panda_joint7"] = j7
        joint_angles["panda_joint8"] = 0.0  # No actuation
        joint_angles["panda_hand_joint"] = 0.0  # No actuation
        joint_angles["panda_finger_joint1"] = 0.05
        joint_angles["panda_finger_joint2"] = 0.05
        joint_angles["panda_grasptarget_hand"] = 0.0
        joint_angles["camera_joint"] = 0.0  # No actuation
        joint_angles["camera_depth_joint"] = 0.0  # No actuation
        joint_angles["camera_depth_optical_joint"] = 0.0  # No actuation
        joint_angles["camera_left_ir_joint"] = 0.0  # No actuation
        joint_angles["camera_left_ir_optical_joint"] = 0.0  # No actuation
        joint_angles["camera_right_ir_joint"] = 0.0  # No actuation
        joint_angles["camera_right_ir_optical_joint"] = 0.0  # No actuation
        joint_angles["camera_color_joint"] = 0.0  # No actuation
        joint_angles["camera_color_optical_joint"] = 0.0  # No actuation

        for joint_index in range(
            pybullet.getNumJoints(robot_id, physicsClientId=physicsClientId)
        ):
            joint_name = pybullet.getJointInfo(
                robot_id, joint_index, physicsClientId=physicsClientId
            )[1].decode("ascii")
            joint_angle = joint_angles.get(joint_name, 0.0)
            # self.physics_client.changeDynamics(robot_id, joint_index, linearDamping=0, angularDamping=0)
            pybullet.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                targetValue=joint_angle,
                physicsClientId=physicsClientId,
            )

    def setup_robot(
        self,
        scene,
        init_q: np.ndarray = None,
    ):
        """This function loads a panda robot to the simulation environment.
        If loading the object fails, the program is stopped and an appropriate get_error message is returned to user.
        Raises:
            ValueError: Unable to load URDF file
        """

        try:
            id = pybullet.loadURDF(
                self.path_to_urdf,
                self.base_position,
                wxyz_to_xyzw(self.base_orientation),
                useFixedBase=1,
                flags=pybullet.URDF_USE_SELF_COLLISION
                | pybullet.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=scene.physics_client_id,
            )

            ik_id = pybullet.loadURDF(
                fileName=self.path_to_urdf,
                basePosition=self.base_position,
                baseOrientation=wxyz_to_xyzw(self.base_orientation),
                useFixedBase=1,
                flags=pybullet.URDF_USE_SELF_COLLISION,
                physicsClientId=scene.ik_client_id,
            )

        except Exception:
            logging.getLogger(__name__).error("Stopping the program")
            raise ValueError(
                "Could not load URDF-file: Check the path to file. Stopping the program."
                "Your path:",
                self.path_to_urdf,
            )
        if init_q is None:
            init_q = (
                3.57795216e-09,
                1.74532920e-01,
                3.30500960e-08,
                -8.72664630e-01,
                -1.14096181e-07,
                1.22173047e00,
                7.85398126e-01,
            )

        robotEndEffectorIndex = 12

        self.client_id = scene.physics_client_id
        self.robot_id = id
        self.robot_id_ik = ik_id
        self.robotEndEffectorIndex = robotEndEffectorIndex
        self.physics_client_id = scene.physics_client_id

        self.jointIndices_with_fingers = [1, 2, 3, 4, 5, 6, 7, 10, 11]
        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=[10, 11],
            controlMode=pybullet.VELOCITY_CONTROL,
            forces=[0.0, 0.0],
        )
        self.jointIndices = [1, 2, 3, 4, 5, 6, 7]
        # enable JointForceTorqueSensor for joints
        # maybe enable joint force sensor for gripper here later!!!
        for jointIndex in self.jointIndices_with_fingers:
            pybullet.enableJointForceTorqueSensor(
                bodyUniqueId=self.robot_id, jointIndex=jointIndex
            )

        if self.pos_ctrl:
            pybullet.setJointMotorControlArray(
                self.robot_id,
                list(np.arange(1, 8)),
                pybullet.VELOCITY_CONTROL,
                forces=list(self.torque_limit),
                physicsClientId=self.physics_client_id,
            )
        else:
            pybullet.setJointMotorControlArray(
                self.robot_id,
                list(np.arange(1, 8)),
                pybullet.VELOCITY_CONTROL,
                forces=list(np.zeros(7)),
                physicsClientId=self.physics_client_id,
            )

        self.receiveState()

        self.beam_to_joint_pos(init_q, run=False)

        scene.state_id = pybullet.saveState()
        self.inhand_cam.set_robot_id(self.robot_id)
