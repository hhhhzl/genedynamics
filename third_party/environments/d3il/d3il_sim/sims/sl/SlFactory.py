import gin

import environments.d3il.d3il_sim.sims.SimFactory as Sims
from environments.d3il.d3il_sim.core import Camera, RobotBase, Scene
from environments.d3il.d3il_sim.sims.sl.SlCamera import SlCamera
from environments.d3il.d3il_sim.sims.sl.SlRobot import SlRobot
from environments.d3il.d3il_sim.sims.sl.SlScene import SlScene
from environments.d3il.d3il_sim.utils.sim_path import d3il_path


class SlFactory(Sims.SimFactory):
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
                "d3il_sim/sims/sl/Config/sl_controller_config.gin"
            )
        gin.parse_config_file(gin_path)
        return SlScene(object_list=object_list, dt=dt, render=render, *args, **kwargs)

    def create_robot(self, scene, *args, **kwargs) -> RobotBase:
        return SlRobot(scene, *args, **kwargs)

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
        return SlCamera(name, width, height, init_pos, init_quat, *args, **kwargs)

    def prim_loading(self):
        return super().prim_loading()


Sims.SimRepository.register(SlFactory(), "sl")
