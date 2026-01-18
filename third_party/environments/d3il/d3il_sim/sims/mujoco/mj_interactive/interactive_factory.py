import environments.d3il.d3il_sim.sims.SimFactory as Sims
from environments.d3il.d3il_sim.core.Robots import RobotBase
from environments.d3il.d3il_sim.sims.mujoco.mj_interactive.ia_robots.ia_mujoco_robot import (
    InteractiveMujocoRobot,
)
from environments.d3il.d3il_sim.sims.mujoco.MujocoFactory import MujocoFactory


class InteractiveMujocoFactory(MujocoFactory):
    def create_robot(self, *args, **kwargs) -> RobotBase:
        return InteractiveMujocoRobot(*args, **kwargs)


Sims.SimRepository.register(InteractiveMujocoFactory(), "mj_interactive")
