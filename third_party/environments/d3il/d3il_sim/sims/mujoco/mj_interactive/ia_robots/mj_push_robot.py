from environments.d3il.d3il_sim.sims.mujoco.MujocoRobot import MujocoRobot
from environments.d3il.d3il_sim.utils.sim_path import d3il_path


class MjPushRobot(MujocoRobot):
    @property
    def xml_file_path(self):
        return d3il_path("./models/mujoco/robots/panda_rod.xml")
