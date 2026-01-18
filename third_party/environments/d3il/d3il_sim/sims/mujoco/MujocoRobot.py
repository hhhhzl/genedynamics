import logging
import xml.etree.ElementTree as Et

import mujoco_py
import numpy as np

from environments.d3il.d3il_sim.controllers import (
    GotoCartPosCartesianRobotController,
    GotoCartPosQuatCartesianRobotController,
)
from environments.d3il.d3il_sim.controllers.Controller import ModelBasedFeedforwardController
from environments.d3il.d3il_sim.controllers.IKControllers import CartPosQuatCartesianRobotController
from environments.d3il.d3il_sim.core import RobotBase, RobotControlInterface
from environments.d3il.d3il_sim.sims.mujoco.mj_utils.mujoco_helpers import reset_mocap2body_xpos
from environments.d3il.d3il_sim.sims.mujoco.MujocoCamera import MjInhandCamera
from environments.d3il.d3il_sim.sims.mujoco.MujocoLoadable import MujocoIncludeTemplate
from environments.d3il.d3il_sim.utils.sim_path import d3il_path


class MujocoRobot(RobotBase, MujocoIncludeTemplate):
    GLOBAL_MJ_ROBOT_COUNTER = 0

    def __init__(
        self,
        scene,
        num_DoF=7,
        base_position=None,
        base_orientation=None,
        gravity_comp=True,
        clip_actions=False,
        xml_path=None,
    ):

        super(MujocoRobot, self).__init__(
            scene, scene.dt, num_DoF, base_position, base_orientation
        )

        self.clip_actions = clip_actions
        self.gravity_comp = gravity_comp

        self.functions = mujoco_py.functions

        self.joint_names = None
        self.joint_indices = None
        self.joint_act_indices = None

        self.gripper_names = None
        self.gripper_indices = None
        self.gripper_act_indices = None

        self.jointTrackingController = ModelBasedFeedforwardController()

        # Global "unique" ID for multibot support
        self._mj_robot_id = MujocoRobot.GLOBAL_MJ_ROBOT_COUNTER
        MujocoRobot.GLOBAL_MJ_ROBOT_COUNTER += 1

        self.inhand_cam = MjInhandCamera(self.add_id2model_key("rgbd"))

        if xml_path is None:
            xml_path = d3il_path("./models/mujoco/robots/panda_rod.xml")
        self._xml_path = xml_path

    def _getJacobian_internal(self, q=None):
        """
        Getter for the jacobian matrix.

        :return: jacobian matrix
        """
        jac = np.zeros((6, 7))
        if q is not None:
            # if we want to have the jacobian for specific joint constelation, do this, otherwise
            # directly read out the jacobian
            # see: http://www.mujoco.org/forum/index.php?threads/inverse-kinematics.3505/

            # first copy current simulation state
            cur_sim_state = self.scene.sim.get_state()
            qpos_idx = self.joint_indices
            self.scene.sim.data.qpos[qpos_idx] = q
            self.functions.mj_kinematics(self.scene.model, self.scene.sim.data)
            self.functions.mj_comPos(self.scene.model, self.scene.sim.data)

        tcp_id = self.add_id2model_key("tcp")
        jac[:3, :] = self.scene.sim.data.get_body_jacp(tcp_id).reshape((3, -1))[
            :, -9:-2
        ]
        jac[3:, :] = self.scene.sim.data.get_body_jacr(tcp_id).reshape((3, -1))[
            :, -9:-2
        ]

        if q is not None:
            # we have to reset the simulation to the state from before
            self.scene.sim.set_state(cur_sim_state)
            # NOTE: Test followin workflow:
            # 		get jacobian for current simulation step
            # 		set qpos to a random position
            # 	    call forward kinematics and compos
            # 	    get new jacobian -> should be different to the one from before
            # 		reset simulation to the state before (with self.scene.sim.set_state
            # 		calculate again jacobian -> should be the same from the state before.
            # 		BUT: IT is not!!! why?
        return jac

    def _getForwardKinematics_internal(self, q=None):
        if q is not None:
            # first copy current simulation state
            cur_sim_state = self.scene.sim.get_state()
            qpos_idx = self.joint_indices
            self.scene.sim.data.qpos[qpos_idx] = q
            self.functions.mj_kinematics(self.scene.model, self.scene.sim.data)

        tcp_id = self.add_id2model_key("tcp")
        cart_pos = self.scene.sim.data.get_body_xpos(tcp_id)
        cart_or = self.scene.sim.data.get_body_xquat(tcp_id)
        if q is not None:
            # reset simulation back to state from before
            self.scene.sim.set_state(cur_sim_state)
        return cart_pos, cart_or

    def prepare_step(self):
        self.command = self.activeController.getControl(self)

        self.preprocessCommand(self.command)
        self.scene.sim.data.ctrl[self.joint_act_indices] = self.uff.copy()[
            : self.num_DoF
        ]
        self.scene.sim.data.ctrl[self.gripper_act_indices] = self.finger_commands
        self.receiveState()

    def receiveState(self):
        if self.joint_names is None:
            self._initialize_joint_names()

        tcp_name = self.add_id2model_key("tcp")

        ### JOINT STATE
        self.current_j_pos = np.array(
            [
                self.scene.sim.data.get_joint_qpos(name) for name in self.joint_names
            ].copy()
        )
        self.current_j_vel = np.array(
            [
                self.scene.sim.data.get_joint_qvel(name) for name in self.joint_names
            ].copy()
        )

        ### ENDEFFECTOR GLOBAL
        self.current_c_pos_global = self.scene.sim.data.get_body_xpos(tcp_name).copy()
        self.current_c_vel_global = self.scene.sim.data.get_body_xvelp(tcp_name).copy()
        self.current_c_quat_global = self.scene.sim.data.get_body_xquat(tcp_name).copy()
        self.current_c_quat_vel_global = np.zeros(4)
        self.current_c_quat_vel_global[1:] = self.scene.sim.data.get_body_xvelr(
            tcp_name
        ).copy()
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

        ### FINGER STATE
        self.current_fing_pos = [
            self.scene.sim.data.get_joint_qpos(j_name) for j_name in self.gripper_names
        ]
        self.current_fing_vel = [
            self.scene.sim.data.get_joint_qvel(j_name) for j_name in self.gripper_names
        ]
        self.gripper_width = self.current_fing_pos[-2] + self.current_fing_pos[-1]

        # self.des_joint_pos = np.zeros((self.num_DoF,)) * np.nan
        # self.des_joint_vel = np.zeros((self.num_DoF,)) * np.nan
        # self.des_joint_acc = np.zeros((self.num_DoF,)) * np.nan

    def get_command_from_inverse_dynamics(self, target_j_acc, mj_calc_inv=False):
        if mj_calc_inv:
            self.scene.sim.data.qacc[
                self.joint_vel_indices + self.gripper_vel_indices
            ] = target_j_acc
            self.functions.mj_inverse(self.scene.model, self.scene.sim.data)
            return self.scene.sim.data.qfrc_inverse[
                self.joint_vel_indices + self.gripper_vel_indices
            ]  # 9 since we have 2 actuations on the fingers
        else:
            return self.scene.sim.data.qfrc_bias[
                self.joint_vel_indices + self.gripper_vel_indices
            ]

    "Do not delete. Might be usefule "

    # def get_pos_for_panda_hand(self, desiredPos, desiredQuat):
    # 	# first set to inital transformation:
    # 	init_transformation = np.linalg.inv(self.scene.sim.data.get_body_xmat('panda_hand'))
    # 	# init positions
    # 	panda_hand_init = init_transformation @ self.current_c_pos
    # 	site_init = init_transformation @ self.scene.sim.data.get_body_xpos('tcp')
    #
    # 	# transform to target orientation
    # 	target_transf_mat = quat2mat(desiredQuat)
    # 	panda_hand_new = target_transf_mat @ panda_hand_init
    # 	site_new = target_transf_mat @ site_init
    #
    # 	# get delta position to site:
    # 	d_pos = desiredPos - site_new
    #
    # 	# add to panda hand new:
    # 	panda_hand_new += d_pos
    # 	return panda_hand_new

    def get_init_qpos(self):
        return np.array(
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

    def _initialize_joint_names(self):
        """Initialize relevant joint and actuator names and indices"""
        ### ARM JOINT NAMES
        self.joint_names = [
            name
            for name in self.scene.sim.model.joint_names
            if name.startswith(self.add_id2model_key("panda_joint"))
        ]

        n_joints = len(self.joint_names)
        assert (
            n_joints == self.num_DoF
        ), "Error, found {} joints, but expected {}".format(n_joints, self.num_DoF)

        ### ENDEFFECTOR JOINT NAMES
        self.gripper_names = [
            name
            for name in self.scene.sim.model.joint_names
            if name.startswith(self.add_id2model_key("panda_finger_joint"))
        ]
        assert (
            len(self.gripper_names) == 2
        ), "Error, found more gripper joints than expected."

        ### ARM INDICES
        self.joint_indices = [
            self.scene.sim.model.get_joint_qpos_addr(joint)
            for joint in self.joint_names
        ]
        self.joint_vel_indices = [
            self.scene.sim.model.get_joint_qvel_addr(joint)
            for joint in self.joint_names
        ]
        self.joint_act_indices = [
            self.scene.sim.model.actuator_name2id(name + "_act")
            for name in self.joint_names
        ]

        ### ENDEFFECTOR ACTUATOR INDICES
        self.gripper_indices = [
            self.scene.sim.model.get_joint_qpos_addr(gripper)
            for gripper in self.gripper_names
        ]
        self.gripper_vel_indices = [
            self.scene.sim.model.get_joint_qvel_addr(gripper)
            for gripper in self.gripper_names
        ]
        self.gripper_act_indices = [
            self.scene.sim.model.actuator_name2id(name + "_act")
            for name in self.gripper_names
        ]

    def set_q(self, joint_pos):
        """
        Sets the value of the robot joints.
        Args:
            joint_pos: Value for the robot joints.

        Returns:
            No return value
        """
        if self.joint_indices is None:
            self._initialize_joint_names()

        qpos = self.scene.sim.data.qpos.copy()
        qpos[self.joint_indices] = joint_pos

        qvel = self.scene.sim.data.qvel.copy()
        # Use len() as qvel somehow can be shorter than qpos??
        qvel[self.joint_vel_indices] = np.zeros(len(self.joint_vel_indices))

        mjSimState = mujoco_py.MjSimState(
            time=0.0, qpos=qpos, qvel=qvel, act=None, udd_state={}
        )
        self.scene.sim.set_state(mjSimState)

    @property
    def xml_file_path(self):
        return self._xml_path

    def add_id2model_key(self, model_key_id: str) -> str:
        """modifies a model key identifier to include the robot id

        Args:
            model_key_id (str): an identifier of the xml model, e.g. joint1

        Returns:
            str: model_key with appended id, e.g. joint1_1
        """
        attrib_split = model_key_id.split("_")
        attrib_split.insert(1, "rb{}".format(self._mj_robot_id))
        attrib_id = "_".join(attrib_split)
        return attrib_id

    def modify_template(self, et: Et.ElementTree) -> str:
        for node in et.iter():
            for attrib in [
                "body1",
                "body2",
                "name",
                "class",
                "childclass",
                "mesh",
                "site",
                "joint",
            ]:
                if attrib in node.attrib:
                    model_key_id = node.get(attrib)
                    attrib_id = self.add_id2model_key(model_key_id)
                    node.set(attrib, attrib_id)

        wb = et.find("worldbody")
        body_root = wb.find("body")
        body_root.set("pos", " ".join(map(str, self.base_position)))
        body_root.set("quat", " ".join(map(str, self.base_orientation)))

        import uuid

        new_path = d3il_path(
            # <<<<<<< HEAD
            "./models/mujoco/robots/panda_tmp_rb{}.xml".format(self._mj_robot_id)
            # =======
            #             "./models/mujoco/robots/panda_tmp_rb{}_{}.xml".format(
            #                 self._mj_robot_id, uuid.uuid1()
            #             )
            # >>>>>>> origin/controller_fixes
        )
        et.write(new_path)
        return new_path


class MujocoMocapRobot(MujocoRobot):
    def __init__(self, scene):

        super(MujocoMocapRobot, self).__init__(scene)

        self.gotoJointController = None

        self.cartesianPosTrackingController = CartPosQuatCartesianRobotController()
        self.cartesianPosQuatTrackingController = CartPosQuatCartesianRobotController()

        self.gotoCartPosController = GotoCartPosCartesianRobotController(self.dt)
        self.gotoCartPosQuatController = GotoCartPosQuatCartesianRobotController(
            self.dt
        )
        self.controlInterface = RobotControlInterface.CartesianInterface
        # self.reset()

    def reset(self):
        """
        Reset robot and scene in mujoco
        """
        super(MujocoMocapRobot, self).reset()

        if self.scene.sim is None:
            return

        if self.scene.sim.model.nmocap > 0 and self.scene.sim.model.eq_data is not None:
            for i in range(self.scene.sim.model.eq_data.shape[0]):
                if self.scene.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    self.scene.sim.model.eq_data[i, :] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    )

        self.scene.sim.forward()

        # Move end effector into position.
        gripper_target = self.scene.sim.data.get_body_xpos("tcp").copy()
        gripper_rotation = np.array([0.0, 1.0, 0.0, 0.0])
        self.scene.sim.data.set_mocap_pos("panda:mocap", gripper_target)
        self.scene.sim.data.set_mocap_quat("panda:mocap", gripper_rotation)
        for _ in range(20):
            self.scene.sim.data.mocap_pos[:] = self.scene.sim.data.get_body_xpos(
                "tcp"
            ).copy()
            self.scene.sim.data.mocap_quat[:] = self.scene.sim.data.get_body_xquat(
                "tcp"
            ).copy()
            self.scene.sim.step()

        self.mocap_setup = True
        self.receiveState()

    def _nextStep(self):

        self.preprocessCommand(self.command)

        reset_mocap2body_xpos(self.scene.sim)
        self.scene.sim.data.mocap_pos[:] = self.command[:3].copy()
        self.scene.sim.data.mocap_quat[:] = self.command[3:].copy()

        gripper_ctrl = self.fing_ctrl_step()
        self.scene.sim.data.ctrl[:] = gripper_ctrl.copy()

        # self.scene.sim.data.ctrl[:] = self.uff.copy()

        # execute simulation for one step
        try:
            self.scene.sim.step()
        except Exception as e:
            logging.getLogger(__name__).error(e)
            logging.getLogger(__name__).error("Simulation step could not be executed")

        self.scene.render()
        self.logData()
        self.receiveState()

        self.step_count += 1
        self.time_stamp += self.dt

    @property
    def xml_file_path(self):
        return d3il_path("./models/mujoco/robots/panda_mocap.xml")
