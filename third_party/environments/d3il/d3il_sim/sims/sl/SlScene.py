from typing import List

import numpy as np

import environments.d3il.d3il_sim.core.sim_object as sim_object
from environments.d3il.d3il_sim.core.Scene import Scene
from environments.d3il.d3il_sim.sims.sl.SlCamera import SlCamera


class SlScene(Scene):
    def __init__(
        self, object_list=None, dt=0.001, skip_home: bool = False, *args, **kwargs
    ):
        """Creates an SLScene.

        Args:
            robot (SLRobot): SLRobot in this scene
            object_list (list, optional): a list of virtual objects in this scene. Defaults to None.
            dt (float, optional): delta time - the control frequency. Defaults to 0.001.
            skip_home (bool, optional): If true, the robot will not be reset to its home position upon scene start / reset. Defaults to False.
        """
        super(SlScene, self).__init__(object_list, dt, *args, **kwargs)

        # Skips go_home command. Important for Teleoperation, as Gravity Compensation is buggy.
        self._skip_home = skip_home

    @property
    def sim_name(self) -> str:
        return "sl"

    def _setup_scene(self):
        # Add Dummy Cameras
        self.inhand_cam = SlCamera("rgbd")
        self.cage_cam = SlCamera("rgbd_cage")
        self.obj_repo.add_object(self.inhand_cam)
        self.obj_repo.add_object(self.cage_cam)
        self._setup_objects(self.obj_repo.get_obj_list())

    ### TODO:MULTIBOT
    def reset(self, obj_pos=None):
        for rb in self.robots:
            rb.reset()
            rb.receiveState()
            rb.jointTrackingController.setSetPoint(rb.current_j_pos)

        for rb in self.robots:
            if not self._skip_home:
                rb.go_home()
            rb.receiveState()

    def _setup_objects(self, sim_objs: List[sim_object.SimObject]):
        for i, obj in enumerate(sim_objs):
            self.obj_repo.register_obj_id(obj, i)

    def load_robot_to_scene(self, robot_init_qpos: np.ndarray = None):
        pass

    def _rt_add_object(self, sim_obj: sim_object.SimObject):
        i = len(self.object_list)
        self.obj_repo.add_object(sim_obj)
        self.obj_repo.register_obj_id(sim_obj, i)

    def _get_obj_pos(self, poi, sim_obj: sim_object.SimObject) -> np.ndarray:
        return [sim_obj.init_pos]

    def _get_obj_quat(self, poi, sim_obj: sim_object.SimObject) -> np.ndarray:
        return [sim_obj.init_quat]

    def render(self):
        pass

    def _sim_step(self):
        pass

    def _remove_object(self, sim_obj: sim_object.SimObject):
        pass

    def _set_obj_pos(self, new_pos, sim_obj: sim_object.SimObject):
        pass

    def _set_obj_pos_and_quat(
        self, new_pos, new_quat, sim_obj: sim_object.SimObject
    ) -> np.ndarray:
        pass

    def _set_obj_quat(self, new_quat, sim_obj: sim_object.SimObject) -> np.ndarray:
        pass

    def _get_obj_seg_id(self, obj_name: str) -> int:
        pass
