import gin

import environments.d3il.d3il_sim.sims.SimFactory as Sims
from environments.d3il.d3il_sim.core import Camera, RobotBase, Scene
from environments.d3il.d3il_sim.utils.sim_path import d3il_path

from .MjCamera import MjCamera
from .MjPrimLoader import mj_load
from .MjRobot import MjRobot
from .MjScene import MjScene


class MjFactory(Sims.SimFactory):
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
        return MjScene(object_list, dt, render, *args, **kwargs)

    def create_robot(self, scene, *args, **kwargs) -> RobotBase:
        return MjRobot(scene, *args, **kwargs)

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
        return MjCamera(name, width, height, init_pos, init_quat)

    def prim_loading(self):
        return mj_load


Sims.SimRepository.register(MjFactory(), "mj_beta")
