import os
import xml.etree.ElementTree as Et
from typing import Tuple

import mujoco
import numpy as np

from environments.d3il.d3il_sim.controllers.Controller import ModelBasedFeedforwardController
from environments.d3il.d3il_sim.core import RobotBase
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import (
    IncludeType,
    get_body_xvelp,
    get_body_xvelr,
)
from environments.d3il.d3il_sim.sims.mj_beta.MjCamera import MjInhandCamera
from environments.d3il.d3il_sim.sims.mj_beta.MjLoadable import (
    MjIncludeTemplate,
    MjXmlLoadable,
)
from environments.d3il.d3il_sim.utils import sim_path


class MjRobot(RobotBase, MjIncludeTemplate):
    GLOBAL_MJ_ROBOT_COUNTER = 0

    def __init__(
        self,
        scene,
        dt=1e-3,
        num_DoF=7,
        base_position=None,
        base_orientation=None,
        gravity_comp=True,
        clip_actions=False,
        root=sim_path.D3IL_DIR,
        xml_path=None,
    ):
        RobotBase.__init__(self, scene, dt, num_DoF, base_position, base_orientation)

        if xml_path is None:
            xml_path = sim_path.d3il_path("./models/mj/robot/panda.xml")
        MjXmlLoadable.__init__(self, xml_path)

        self.clip_actions = clip_actions
        self.gravity_comp = gravity_comp

        self.joint_names = None
        self.joint_indices = None
        self.joint_act_indices = None

        self.gripper_names = None
        self.gripper_indices = None
        self.gripper_act_indices = None

        self.jointTrackingController = ModelBasedFeedforwardController()

        # Global "unique" ID for multibot support
        self._mj_robot_id = MjRobot.GLOBAL_MJ_ROBOT_COUNTER
        MjRobot.GLOBAL_MJ_ROBOT_COUNTER += 1
        self.root = root

        self.inhand_cam = MjInhandCamera(self.add_id2model_key("rgbd"))

        # Position and velocity for freezing operations
        self.qpos_freeze = None
        self.qvel_freeze = None
        self.ctrl_freeze = None

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
            mujoco.mj_kinematics(self.scene.model, self.scene.sim.data)
            mujoco.mj_comPos(self.scene.model, self.scene.sim.data)

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
            self.scene.data.qpos[qpos_idx] = q
            mujoco.mj_kinematics(self.scene.model, self.scene.data)

        tcp_id = self.add_id2model_key("tcp")
        cart_pos = self.scene.data.get_body_xpos(tcp_id)
        cart_or = self.scene.data.get_body_xquat(tcp_id)
        if q is not None:
            # reset simulation back to state from before
            self.scene.sim.set_state(cur_sim_state)
        return cart_pos, cart_or

    def prepare_step(self):
        self.command = self.activeController.getControl(self)

        self.preprocessCommand(self.command)
        self.scene.data.ctrl[self.joint_act_indices] = self.uff.copy()[: self.num_DoF]
        self.scene.data.ctrl[self.gripper_act_indices] = self.finger_commands
        self.receiveState()

    def receiveState(self):
        if self.joint_indices is None:
            self._init_jnt_indices()

        tcp_name = self.add_id2model_key("tcp")

        ### JOINT STATE
        self.current_j_pos = np.array(
            [self.scene.data.joint(name).qpos.copy() for name in self.joint_names]
        ).squeeze()
        self.current_j_vel = np.array(
            [self.scene.data.joint(name).qvel.copy() for name in self.joint_names]
        ).squeeze()

        test = self.scene.data.body(tcp_name)
        ### ENDEFFECTOR GLOBAL
        self.current_c_pos_global = self.scene.data.body(tcp_name).xpos.copy()
        self.current_c_vel_global = get_body_xvelp(
            self.scene.model,
            self.scene.data,
            tcp_name,
        )
        self.current_c_quat_global = self.scene.data.body(tcp_name).xquat.copy()
        self.current_c_quat_vel_global = np.zeros(4)

        self.current_c_quat_vel_global[1:] = get_body_xvelr(
            self.scene.model, self.scene.data, tcp_name
        )
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
        self.current_fing_pos = np.array(
            [self.scene.data.joint(name).qpos.copy() for name in self.gripper_names]
        ).squeeze()
        self.current_fing_vel = np.array(
            [self.scene.data.joint(name).qvel.copy() for name in self.gripper_names]
        ).squeeze()
        self.gripper_width = self.current_fing_pos[-2] + self.current_fing_pos[-1]

    def get_command_from_inverse_dynamics(self, target_j_acc, mj_calc_inv=False):
        if mj_calc_inv:
            self.scene.data.qacc[
                self.joint_vel_indices + self.gripper_vel_indices
            ] = target_j_acc
            mujoco.mj_inverse(self.scene.model, self.scene.sim.data)
            return self.scene.data.qfrc_inverse[
                self.joint_vel_indices + self.gripper_vel_indices
            ]  # 9 since we have 2 actuations on the fingers
        else:
            return self.scene.data.qfrc_bias[
                self.joint_vel_indices + self.gripper_vel_indices
            ]

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

    def _init_jnt_indices(self):
        """Initialize relevant joint and actuator names and indices"""
        n_joints = len(self.joint_names)
        assert (
            n_joints == self.num_DoF
        ), "Error, found {} joints, but expected {}".format(n_joints, self.num_DoF)
        assert (
            len(self.gripper_names) == 2
        ), "Error, found more gripper joints than expected."

        self.joint_indices = []
        self.joint_vel_indices = []
        self.gripper_indices = []
        self.gripper_vel_indices = []
        for jnt_name in self.joint_names:
            jnt_id = mujoco.mj_name2id(
                self.scene.model, type=mujoco.mjtObj.mjOBJ_JOINT, name=jnt_name
            )
            self.joint_indices.append(self.scene.model.jnt_qposadr[jnt_id])
            self.joint_vel_indices.append(self.scene.model.jnt_dofadr[jnt_id])
        for grp_name in self.gripper_names:
            grp_id = mujoco.mj_name2id(
                self.scene.model, type=mujoco.mjtObj.mjOBJ_JOINT, name=grp_name
            )
            self.gripper_indices.append(self.scene.model.jnt_qposadr[grp_id])
            self.gripper_vel_indices.append(self.scene.model.jnt_dofadr[grp_id])
        self.joint_act_indices = [
            mujoco.mj_name2id(
                self.scene.model, type=mujoco.mjtObj.mjOBJ_ACTUATOR, name=name + "_act"
            )
            for name in self.joint_names
        ]
        self.gripper_act_indices = [
            mujoco.mj_name2id(
                self.scene.model, type=mujoco.mjtObj.mjOBJ_ACTUATOR, name=name + "_act"
            )
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
            self._init_jnt_indices()

        qpos = self.scene.data.qpos.copy()
        qpos[self.joint_indices] = joint_pos

        qvel = self.scene.data.qvel.copy()
        # Use len() as qvel somehow can be shorter than qpos??
        qvel[self.joint_vel_indices] = np.zeros(len(self.joint_vel_indices))
        self.scene.set_state(qpos, qvel)

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
        self.joint_names = None
        self.joint_indices = None
        self.joint_act_indices = None

        self.gripper_names = None
        self.gripper_indices = None
        self.gripper_act_indices = None

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

        self.joint_names = []
        self.gripper_names = []
        for jnt in wb.iter("joint"):
            if jnt.get("name") is not None and jnt.get("name").startswith(
                self.add_id2model_key("panda_joint")
            ):
                self.joint_names.append(jnt.get("name"))
            if jnt.get("name") is not None and jnt.get("name").startswith(
                self.add_id2model_key("panda_finger_joint")
            ):
                self.gripper_names.append(jnt.get("name"))

        import uuid

        new_path = sim_path.d3il_path(
            f"./models/mj/robot/panda_tmp_rb{self._mj_robot_id}_{uuid.uuid1()}.xml"
        )
        et.write(new_path)
        return new_path
