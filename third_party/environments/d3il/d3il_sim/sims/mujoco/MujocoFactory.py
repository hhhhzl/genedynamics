import gin

import environments.d3il.d3il_sim.sims.SimFactory as Sims
from environments.d3il.d3il_sim.core import Camera, RobotBase, Scene
from environments.d3il.d3il_sim.utils.sim_path import d3il_path

from .MujocoCamera import MujocoCamera
from .MujocoPrimLoader import to_mj_xml
from .MujocoRobot import MujocoMocapRobot, MujocoRobot
from .MujocoScene import MujocoScene


class MujocoFactory(Sims.SimFactory):
    def create_scene(
        self,
        gin_path=None,
        object_list: list = None,
        dt: float = 0.001,
        render: Scene.RenderMode = Scene.RenderMode.HUMAN,
        *args,
        **kwargs
    ) -> Scene:

        if gin_path is None:
            gin_path = d3il_path(
                "d3il_sim/controllers/Config/mujoco_controller_config.gin"
            )
        gin.parse_config_file(gin_path)
        return MujocoScene(object_list, dt, render, *args, **kwargs)

    def create_robot(self, scene, *args, **kwargs) -> RobotBase:
        return MujocoRobot(scene, *args, **kwargs)

    def create_camera(
        self,
        name: str,
        width: int = 1000,
        height: int = 1000,
        init_pos=None,
        init_quat=None,
        *args,
        **kwargs
    ) -> Camera:
        return MujocoCamera(name, width, height, init_pos, init_quat)

    def prim_loading(self):
        return to_mj_xml


Sims.SimRepository.register(MujocoFactory(), "mujoco")


class MujocoMocapFactory(MujocoFactory):
    def create_robot(self, scene, *args, **kwargs) -> RobotBase:
        return MujocoMocapRobot(scene)


Sims.SimRepository.register(MujocoMocapFactory(), "mujoco_mocap")
