import copy
import logging
from typing import Union

import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient

from environments.d3il.d3il_sim.core import Scene
from environments.d3il.d3il_sim.core.sim_object import IntelligentSimObject, SimObject
from environments.d3il.d3il_sim.sims.pybullet.PybulletCamera import PbCageCam
from environments.d3il.d3il_sim.sims.pybullet.PybulletLoadable import PybulletLoadable
from environments.d3il.d3il_sim.utils.geometric_transformation import wxyz_to_xyzw, xyzw_to_wxyz
from environments.d3il.d3il_sim.utils.sim_path import d3il_path

PYBULLET = "pybullet"

"""
IMPORTANT INFORMATION
PyBullet does use XYZW for Quaternions, We use WXYZ as standard.
For all interfaces we therefore have to convert the quaternions to the appropriate format
"""


class PyBulletScene(Scene):
    """
    This class allows to build a scene for the robot simulation.
    The standard scene is a model of the Panda robot on a
    table. The returned ids of the assets are saved as an
    attribute to the object of the Panda_Robot.
    The .urdf files which contain the scene assets (e.g. cubes etc.)
    are saved in the 'envs' folder of the project.
    """

    def __init__(
        self,
        object_list=None,
        dt=0.001,
        render=Scene.RenderMode.HUMAN,
        realtime=False,
        *args,
        **kwargs
    ):
        if object_list is not None:
            for object in object_list:
                object.init_quat = wxyz_to_xyzw(object.init_quat)

        super(PyBulletScene, self).__init__(
            object_list=object_list, dt=dt, render=render, *args, **kwargs
        )

        # Pybullet Realtime
        self.realtime = realtime
        if self.realtime:
            p.setRealTimeSimulation(1)

        # Connect with simulator
        if self.render_mode == Scene.RenderMode.HUMAN:
            self.physics_client = BulletClient(p.GUI)
        else:
            self.physics_client = BulletClient(p.DIRECT)
        self.physics_client_id = self.physics_client._client

        self.ik_client = BulletClient(connection_mode=p.DIRECT)
        self.ik_client_id = self.ik_client._client

        p.setPhysicsEngineParameter(enableFileCaching=0)

        # Already add the cagecam so that it can be configured by the user before starting the scene
        self.cage_cam = PbCageCam()

    @property
    def sim_name(self) -> str:
        return "pybullet"

    def _setup_scene(self):
        # load surroundings
        p.loadURDF(
            d3il_path("./models/pybullet/objects/plane/plane.urdf"),
            physicsClientId=self.physics_client_id,
            basePosition=[0, 0, -0.94],
        )
        p.loadURDF(
            d3il_path("./models/pybullet/objects/plane/plane.urdf"),
            physicsClientId=self.ik_client_id,
            basePosition=[0, 0, -0.94],
        )

        table_start_position = [0.2, 0, -0.02]
        table_start_orientation = [0.0, 0.0, 0.0]
        table_start_orientation_quat = p.getQuaternionFromEuler(table_start_orientation)
        p.loadURDF(
            d3il_path(
                "./models/pybullet/surroundings/lab_surrounding.urdf"
            ),
            table_start_position,
            table_start_orientation_quat,
            useFixedBase=1,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.physics_client_id,
        )

        # set Physics
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.81)

        self.add_object(self.cage_cam)

        for rb in self.robots:
            self.add_object(rb.inhand_cam)

        self._setup_objects(self.obj_repo.get_obj_list())

    def add_object(self, sim_obj: SimObject):
        sim_obj.init_quat = wxyz_to_xyzw(sim_obj.init_quat)
        return super().add_object(sim_obj)

    def load_robot_to_scene(self, robot_init_qpos):
        if robot_init_qpos is not None:
            if robot_init_qpos.ndim == 1:
                robot_init_qpos = np.expand_dims(robot_init_qpos, 0)
        else:
            robot_init_qpos = [None] * len(self.robots)

        for i in range(len(self.robots)):
            self.robots[i].setup_robot(self, robot_init_qpos[i])

    def reset(self, obj_pos=None):
        if obj_pos is None:
            obj_pos = []

        p.restoreState(self.state_id)

        for rb in self.robots:
            rb.reset()
            rb.receiveState()

        for (obj, new_pos) in obj_pos:
            self.set_obj_pos(new_pos, obj)

    def _get_obj_seg_id(self, obj_name: str):
        """
        Returns the ID of an Object based on an obj_name
        This ID is the one used in the Segmentation Image retrievable through get_segmentation
        :param obj_name
        """
        return self.obj_repo.get_id_from_name(obj_name)

    def _pb_load_obj(self, sim_obj: Union[SimObject, PybulletLoadable]):
        obj_id = sim_obj.pb_load(self.physics_client_id)
        self.obj_repo.register_obj_id(sim_obj, obj_id)

        if isinstance(sim_obj, IntelligentSimObject):
            sim_obj.register_sim(self.physics_client_id, self.sim_name)

    def _setup_objects(self, object_list):
        for obj in object_list:
            self._pb_load_obj(obj)

    def _rt_add_object(self, sim_obj: SimObject):
        self.obj_repo.add_object(sim_obj)
        self._pb_load_obj(sim_obj)

    def _get_obj_pos(self, poi, sim_obj: SimObject):
        pos, _ = p.getBasePositionAndOrientation(
            bodyUniqueId=sim_obj.obj_id, physicsClientId=self.physics_client_id
        )
        return np.asarray(pos)

    def _get_obj_quat(self, poi, sim_obj: SimObject):
        _, quat = p.getBasePositionAndOrientation(
            bodyUniqueId=sim_obj.obj_id, physicsClientId=self.physics_client_id
        )
        return xyzw_to_wxyz(quat)

    def _set_obj_pos(self, new_pos, sim_obj):
        self._set_obj_pos_and_quat(new_pos, None, sim_obj)

    def _set_obj_quat(self, new_quat, sim_obj):
        self._set_obj_pos_and_quat(None, new_quat, sim_obj)

    def _set_obj_pos_and_quat(self, new_pos, new_quat, sim_obj: SimObject):
        if new_pos is None and new_quat is None:
            logging.getLogger(__name__).warning(
                "Expected at least either a new position or quaternion for set_obj_pos_and_quat"
            )
            return

        o_pos, o_quat = p.getBasePositionAndOrientation(
            bodyUniqueId=sim_obj.obj_id, physicsClientId=self.physics_client_id
        )

        if new_pos is not None:
            o_pos = new_pos
        if new_quat is not None:
            o_quat = wxyz_to_xyzw(new_quat)

        p.resetBasePositionAndOrientation(
            bodyUniqueId=sim_obj.obj_id, posObj=o_pos, ornObj=o_quat
        )

    def _remove_object(self, sim_obj: SimObject):
        p.removeBody(sim_obj.obj_id)

    def render(self):
        # Pybullet renders internally. Scene can pass.
        pass

    def _sim_step(self):
        p.stepSimulation()
