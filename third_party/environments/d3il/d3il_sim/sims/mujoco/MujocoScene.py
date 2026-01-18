import logging
from typing import List

import numpy as np
from mujoco_py import MjSim, MjSimState

from environments.d3il.d3il_sim.core.Scene import Scene
from environments.d3il.d3il_sim.core.sim_object.sim_object import IntelligentSimObject, SimObject
from environments.d3il.d3il_sim.sims.mujoco.mj_utils.mujoco_scene_object import MujocoSurrounding
from environments.d3il.d3il_sim.sims.mujoco.mj_utils.mujoco_scene_parser import MujocoSceneParser
from environments.d3il.d3il_sim.sims.mujoco.mj_utils.mujoco_viewer import MujocoViewer
from environments.d3il.d3il_sim.sims.mujoco.MujocoCamera import MjCageCam
from environments.d3il.d3il_sim.utils import sim_path


class MujocoScene(Scene):
    def __init__(
        self,
        object_list=None,
        dt=0.001,
        render=Scene.RenderMode.HUMAN,
        surrounding=None,
        random_env=False,
        proc_id="",
        main_xml_path=None,
        *args,
        **kwargs,
    ):

        super(MujocoScene, self).__init__(
            object_list=object_list, dt=dt, render=render, *args, **kwargs
        )

        self.sim: MjSim = None
        self.model = None
        self.viewer = None

        self.init_qpos, self.init_qvel = None, None
        self.random_env = random_env

        if surrounding is None:
            surrounding = sim_path.d3il_path(
                "./models/mujoco/surroundings/lab_surrounding.xml"
            )

        self.surrounding = MujocoSurrounding(surrounding)

        self.mj_scene_parser = MujocoSceneParser(
            proc_id=proc_id, main_xml_path=main_xml_path
        )

        self.cage_cam = MjCageCam()
        self.add_object(self.cage_cam)

    @property
    def sim_name(self) -> str:
        return "mujoco"

    def _setup_scene(self):
        self.sim, self.model = self.mj_scene_parser.create_scene(
            self.robots, self.surrounding, self.obj_repo.get_obj_list(), self.dt
        )

        self.viewer = MujocoViewer(self.sim, rm=self.render_mode.value)  # Renderer

        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = np.zeros(self.sim.data.qvel.shape)

        for rb in self.robots:
            self.add_object(rb.inhand_cam)

        self._setup_objects(self.obj_repo.get_obj_list())

    def load_robot_to_scene(self, robot_init_qpos: np.ndarray = None):
        """
        Sets the initial joint position of the panda robot.

        Args:
            robot_init_qpos: numpy array (num dof,); initial joint positions

        Returns:
            No return value
        """
        if robot_init_qpos is None:
            robot_init_qpos = np.stack([robot.get_init_qpos() for robot in self.robots])
        else:
            robot_init_qpos = np.asarray(robot_init_qpos)
        # Check input dimensionality, in case of legacy code with only one robot
        if robot_init_qpos.ndim == 1:
            robot_init_qpos = np.expand_dims(robot_init_qpos, 0)

        for i in range(len(self.robots)):
            self.robots[i].beam_to_joint_pos(robot_init_qpos[i], run=False)
        self.init_qpos = self.sim.data.qpos.copy()

    def _sim_step(self):
        self.sim.step()

    def render(self):
        self.viewer.render()

    def reset(self, obj_pos=None):
        """Resets the scene (including the robot) to the initial conditions."""
        if obj_pos is None:
            obj_pos = []

        for rb in self.robots:
            rb.reset()

        self.sim.reset()
        # Set initial position and velocity
        qpos = self.sim.data.qpos.copy()
        qpos[:] = self.init_qpos

        qvel = np.zeros(self.sim.data.qvel.shape)
        mjSimState = MjSimState(time=0.0, qpos=qpos, qvel=qvel, act=None, udd_state={})
        self.sim.set_state(mjSimState)

        for (obj, new_pos) in obj_pos:
            self.set_obj_pos(new_pos, obj)

        self.sim.forward()

        for rb in self.robots:
            rb.receiveState()

    def start_recording(self, nframes):
        self.viewer.start_recording(nframes=nframes)

    def _setup_objects(self, sim_objs: List[SimObject]):
        for i, obj in enumerate(sim_objs):
            self.obj_repo.register_obj_id(obj, i)

            if isinstance(obj, IntelligentSimObject):
                obj.register_sim(self.sim, self.sim_name)

    def _rt_add_object(self, sim_obj: SimObject):
        raise RuntimeError(
            "Adding objects in MuJoCo only possible prior to scene setup."
        )

    def _get_obj_seg_id(self, obj_name: str):
        """
        Returns the ID of an Object based on an obj_name
        This ID is the one used in the Segmentation Image retrievable through get_segmentation
        :param obj_name
        """
        return self.sim.model.geom_name2id(obj_name)

    def _get_obj_pos(self, poi, sim_obj: SimObject):
        return self.sim.data.get_body_xpos(poi).copy()

    def _get_obj_quat(self, poi, sim_obj: SimObject):
        return self.sim.data.get_body_xquat(poi).copy()

    def _set_obj_pos(self, new_pos, sim_obj: SimObject):
        self._set_obj_pos_and_quat(new_pos, None, sim_obj=sim_obj)

    def _set_obj_quat(self, new_quat, sim_obj: SimObject):
        self._set_obj_pos_and_quat(None, new_quat, sim_obj=sim_obj)

    def _set_obj_pos_and_quat(self, new_pos, new_quat, sim_obj: SimObject):
        if new_pos is None and new_quat is None:
            logging.getLogger(__name__).warning(
                "Expected at least either a new position or quaternion for set_obj_pos_and_quat"
            )
            return

        body_id = self.model.body_name2id(sim_obj.name)
        body_jnt_addr = self.model.body_jntadr[body_id]
        qposadr = self.model.jnt_qposadr[body_jnt_addr]

        if new_pos is not None:
            assert len(new_pos) == 3, print(
                f"Expected a positions list of 3 values, got {len(new_pos)}"
            )

            if body_jnt_addr == -1:
                # Static object
                self.model.body_pos[body_id] = new_pos
            else:
                # Object with joint
                self.sim.data.qpos[qposadr : qposadr + 3] = new_pos

        if new_quat is not None:
            assert len(new_quat) == 4, print(
                f"Expected a quaternions list of 4 values, got {len(new_quat)}"
            )
            if body_jnt_addr == -1:
                # Static object
                self.model.body_quat[body_id] = new_quat
            else:
                # Object with joint
                self.sim.data.qpos[qposadr + 3 : qposadr + 7] = new_quat

    def _remove_object(self, sim_obj: SimObject):
        raise RuntimeError(
            "Removing objects in MuJoCo only possible prior to scene setup."
        )
