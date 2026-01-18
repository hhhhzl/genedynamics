import enum
from abc import ABC, abstractmethod
from typing import List

import numpy as np

import environments.d3il.d3il_sim.core.sim_object as sim_object
import environments.d3il.d3il_sim.core.time_keeper as time_keeper
import environments.d3il.d3il_sim.utils.geometric_transformation as geom_trans
from environments.d3il.d3il_sim.core import Camera, Robots


class Scene(ABC):
    """
    This class allows to build a scene for the robot simulation. The standard scene is a model of
    the Panda robot with surroundings. The scene manages access to the loaded objects and can be
    used to query for object positions and orientations
    """

    class RenderMode(enum.Enum):
        BLIND = "blind"
        OFFSCREEN = "offscreen"
        HUMAN = "human"

    def __init__(
        self,
        object_list=None,
        dt=0.001,
        render: RenderMode = RenderMode.HUMAN,
        *args,
        **kwargs
    ):
        if object_list is None:
            object_list = []

        self.obj_repo = sim_object.SimObjectRepository(object_list)

        self.dt = dt
        self.render_mode = render

        self.setup_done = False

        self._robots: List[Robots.RobotBase] = []
        self.inhand_cam = None
        self.cage_cam = None

        self.step_callbacks = []
        self.additional_loggers = []
        self.time_keeper = time_keeper.TimeKeeper(self.dt)

    def add_robot(self, robot):
        self._robots.append(robot)

    @property
    def robots(self):
        return self._robots

    def register_callback(self, fn, **kwargs):
        """register a function for callback after a simulation step
        The callback function is passed the robot. Additional keyword arguments
        can also be registered.

        Args:
            fn (function): _description_
        """
        self.step_callbacks.append((fn, kwargs))

    @property
    @abstractmethod
    def sim_name(self) -> str:
        """the simulator name,
        for registering an IntelligentSimObject

        Returns:
            str: name of the simulator
        """
        return "NONE"

    @abstractmethod
    def _setup_scene(self):
        """abstract method. called first in the start() function.
        This metod should load all objects and instantiate them, except for the Robot,
        which gets loaded afterwards in load_robot_to_scene()
        Remember to call self._setup_objects() in this function,
        to finalize the object creation such as obj_id assignment.
        """
        self._setup_objects(self.object_list)

    @abstractmethod
    def reset(self, obj_pos=None):
        """resets the scene and all its contained objects to their initial state

        Args:
            obj_pos (optional): Can be a datastructure defining specific positions for the objects.
            Defaults to None.
        """
        raise NotImplementedError

    def start(self, robot_init_qpos: np.ndarray = None):
        """starts the scene
        all objects and the robot gets loaded and initialized.
        Must be called once to start the scene.

        Args:
            robot_init_qpos (np.ndarray, optional): initial robot joint configuration.
                Shape (Num_robot, Num_joints). Defaults to None.
        """
        self._setup_scene()
        self.load_robot_to_scene(robot_init_qpos=robot_init_qpos)
        self.setup_done = True
        self.reset()

    @abstractmethod
    def render(self):
        """render procedure for the scene.
        As this is highly dependent on the used simulator, it must be implemented by
        each concrete child.
        """
        raise NotImplementedError

    def next_step(self, log=True):
        for rb in self.robots:
            rb.prepare_step()

        self._sim_step()

        self.time_keeper.tick()

        for rb in self.robots:
            rb.tick()
            rb.receiveState()

        for call_back, kwargs in self.step_callbacks:
            call_back(**kwargs)
        if log:
            self.log_data()

        self.render()

    @abstractmethod
    def _sim_step(self):
        raise NotImplementedError

    @abstractmethod
    def load_robot_to_scene(self, robot_init_qpos: np.ndarray = None):
        """internal function loading the robot into the scene

        Args:
            robot_init_qpos (np.ndarray, optional): initial robot joint configuration.
                Shape (Num_robot, Num_joints). Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_objects(self, sim_objs: List[sim_object.SimObject]):
        """abstract method to finish loading the SimObjects into the scene.
        This should contain the steps to add all SimObjects to the internal

        Args:
            sim_objs (List[SimObject]): list of SimObjects to process
        """
        raise NotImplementedError

    @abstractmethod
    def _rt_add_object(self, sim_obj: sim_object.SimObject):
        """abstract internal function for adding objects AFTER the scene has been started.
        Some simulations such as Mujoco do not support this and will throw an error

        Args:
            sim_obj (SimObject): the SimObject to load into the scene
        """
        raise NotImplementedError

    def add_object(self, sim_obj: sim_object.SimObject):
        """adds an object to the scene.
        Supports both adding Objects before and after start().

        Args:
            sim_obj (SimObject): The SimObject.
        """
        if not self.setup_done:
            self.obj_repo.add_object(sim_obj)
        else:
            self._rt_add_object(sim_obj)

    def list_objects(self):
        return list(self._objects.keys())

    def get_object(self, name: str = None, obj_id: int = None) -> sim_object.SimObject:
        """getter to retrieve a SimObject that has been added to the Scene.

        Args:
            name (str, optional): the SimObject's name. Defaults to None.
            obj_id (int, optional): the SimObject's id. Defaults to None.

        Returns:
            SimObject
        """
        return self.obj_repo.get_object(name, obj_id)

    def _query_pois(self, sim_obj: sim_object.SimObject, fn) -> np.ndarray:
        """private function. uses the fn callback to read the SimObject's
        POI data from the simulation state.

        Args:
            sim_obj (SimObject): the SimObject
            fn (function): a function

        Returns:
            np.ndarray: an array with information for each poi
        """
        if len(sim_obj.get_poi()) > 1:
            return np.array([fn(poi, sim_obj) for poi in sim_obj.get_poi()])
        else:
            return fn(sim_obj.get_poi()[0], sim_obj)

    @abstractmethod
    def _get_obj_seg_id(self, obj_name: str) -> int:
        """
        abstract private function. Returns the ID of an Object based on an obj_name
        This ID is the one used in the Segmentation Image retrievable through get_segmentation
        Args:
            obj_name: The name of the object

        Returns:
            int: ID of the object as used in the segmentation images

        """
        raise NotImplementedError

    def get_obj_seg_id(
        self,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ) -> int:
        """
        Returns the ID of an Object based on an obj_name
        This ID is the one used in the Segmentation Image retrievable through get_segmentation
        Args:
            sim_obj (SimObject, optional): A SimObject. Defaults to None.
            obj_name (str, optional): A SimObject's name. Defaults to None.
            obj_id (int, optional): A SimObject's ID. Defaults to None.

        Returns:
            int: ID of the object as used in the segmentation images
        """
        if obj_name is None:
            if sim_obj is None:
                sim_obj = self.get_object(obj_id=obj_id)
            obj_name = sim_obj.name

        return self._get_obj_seg_id(obj_name=obj_name)

    @abstractmethod
    def _get_obj_pos(self, poi, sim_obj: sim_object.SimObject) -> np.ndarray:
        """abstract private function. read the XYZ position for a SimObject's
        POI from the Simulation.

        Args:
            poi: a SimObject's Point of Interest (POI). Might be Simulator specific
            sim_obj (SimObject): the SimObject the POI belongs to

        Returns:
            np.ndarray: XYZ coordinates
        """
        raise NotImplementedError

    def get_obj_pos(
        self,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ) -> np.ndarray:
        """reads the XYZ positions for all POIs of a SimObject.
        The SimObject can be defined by its object instance, its name or its object id.

        Args:
            sim_obj (SimObject, optional): A SimObject. Defaults to None.
            obj_name (str, optional): A SimObject's name. Defaults to None.
            obj_id (int, optional): A SimObject's ID. Defaults to None.

        Returns:
            np.ndarray: A Numpy Array with XYZ coordinates for each POI. Has the Shape (n, 3)
        """
        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)
        return self._query_pois(sim_obj, self._get_obj_pos)

    @abstractmethod
    def _get_obj_quat(self, poi, sim_obj: sim_object.SimObject) -> np.ndarray:
        """abstract private function. read the WXYZ quaternion for a SimObject's
        POI from the Simulation.

        Args:
            poi: a SimObject's Point of Interest (POI). Might be Simulator specific
            sim_obj (SimObject): the SimObject the POI belongs to

        Returns:
            np.ndarray: WXYZ Quaternion
        """
        raise NotImplementedError

    def get_obj_quat(
        self,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ) -> np.ndarray:
        """reads the WXYZ quaternion orientation for all POIs of a SimObject.
        The SimObject can be defined by its object instance, its name or its object id.

        Args:
            sim_obj (SimObject, optional): A SimObject. Defaults to None.
            obj_name (str, optional): A SimObject's name. Defaults to None.
            obj_id (int, optional): A SimObject's ID. Defaults to None.

        Returns:
            np.ndarray: A Numpy Array with WXYZ quaternions for each POI. Has the Shape (n, 4)
        """
        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)
        return self._query_pois(sim_obj, self._get_obj_quat)

    def _get_obj_rot_mat(self, poi, sim_obj: sim_object.SimObject) -> np.ndarray:
        """private function. read the  quaternion for a SimObject's
        POI from the Simulation.
        Args:
            poi: a SimObject's Point of Interest (POI). Might be Simulator specific
            sim_obj (SimObject): the SimObject the POI belongs to
        Returns:
            np.ndarray: WXYZ Quaternion
        """
        quat = self._get_obj_quat(poi, sim_obj)
        return geom_trans.mat2quat(quat)

    def get_obj_rot_mat(
        self,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ) -> np.ndarray:
        """reads the rotation matrix for all POIs of a SimObject.
        The SimObject can be defined by its object instance, its name or its object id.
        Args:
            sim_obj (SimObject, optional): A SimObject. Defaults to None.
            obj_name (str, optional): A SimObject's name. Defaults to None.
            obj_id (int, optional): A SimObject's ID. Defaults to None.
        Returns:
            np.ndarray: A Numpy Array with WXYZ quaternions for each POI. Has the Shape (n, 4)
        """
        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)
        return self._query_pois(sim_obj, self._get_obj_rot_mat)

    @abstractmethod
    def _set_obj_pos(self, new_pos, sim_obj: sim_object.SimObject):
        """abstract private function. sets the Position for a SimObject's
        Args:
            new_pos: [x, y, z] a SimObject's new position
            sim_obj (SimObject): the SimObject the new Pose belongs to
        """
        raise NotImplementedError

    def set_obj_pos(
        self,
        new_pos,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ):
        """Sets the Position of a specified Object
        The SimObject can be defined by its object instance, its name or its object id.
        Args:
            new_pos ([x,y,z]): the position the sim_obj is set to.
            sim_obj (SimObject, optional): A SimObject. Defaults to None.
            obj_name (str, optional): A SimObject's name. Defaults to None.
            obj_id (int, optional): A SimObject's ID. Defaults to None.
        """
        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)

        self._set_obj_pos(new_pos, sim_obj)

    @abstractmethod
    def _set_obj_quat(self, new_quat, sim_obj: sim_object.SimObject) -> np.ndarray:
        """abstract private function. sets the Quaternion for a SimObject's
        Args:
            new_quat: [w,x,y,z] a SimObject's new Rotation
            sim_obj (SimObject): the SimObject the new Pose belongs to
        """
        raise NotImplementedError

    def set_obj_quat(
        self,
        new_quat,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ):
        """Sets the Quaternion/Rotation of a specified Object
        The SimObject can be defined by its object instance, its name or its object id.
        Args:
            new_quat ([w,x,y,z], optional): [description]. Defaults to None.
            sim_obj (SimObject, optional): A SimObject. Defaults to None.
            obj_name (str, optional): A SimObject's name. Defaults to None.
            obj_id (int, optional): A SimObject's ID. Defaults to None.
        """
        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)

        self._set_obj_quat(new_quat, sim_obj)

    @abstractmethod
    def _set_obj_pos_and_quat(
        self, new_pos, new_quat, sim_obj: sim_object.SimObject
    ) -> np.ndarray:
        """abstract private function. sets the Position and Quaternion for a SimObject
        Args:
            new_pos: [x,y,z] a SimObject's new Position
            new_quat: [w,x,y,z] a SimObject's new Quaternion
            sim_obj (SimObject): the SimObject the new Pose belongs to
        """
        raise NotImplementedError

    def set_obj_pos_and_quat(
        self,
        new_pos,
        new_quat,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ):
        """Sets the Position and QUaternion of a specified Object
        The SimObject can be defined by its object instance, its name or its object id.
        Args:
            new_pos ([x,y,z]): the position the sim_obj is set to.
            new_quat ([w,x,y,z], optional): the quaternion the sim_obj is set to.
            sim_obj (SimObject, optional): A SimObject. Defaults to None.
            obj_name (str, optional): A SimObject's name. Defaults to None.
            obj_id (int, optional): A SimObject's ID. Defaults to None.
        """
        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)

        self._set_obj_pos_and_quat(new_pos, new_quat, sim_obj)

    @abstractmethod
    def _remove_object(self, sim_obj: sim_object.SimObject):
        raise NotImplementedError

    def remove_object(
        self,
        sim_obj: sim_object.SimObject = None,
        obj_name: str = None,
        obj_id: int = None,
    ):
        if sim_obj is None:
            sim_obj = self.get_object(obj_name, obj_id)

        self._remove_object(sim_obj)
        self.obj_repo.remove_object(sim_obj)

    def get_cage_cam(self) -> Camera.Camera:
        """getter for the cage mounted camera"""
        return self.cage_cam

    def start_logging(self, duration: float = 300.0, **kwargs):
        """
        Start all Loggers in the scene.
        """
        for rb in self.robots:
            rb.start_logging(duration, **kwargs)

        for logger in self.additional_loggers:
            logger.start_logging(duration, **kwargs)

    def stop_logging(self):
        """
        Stop all Loggers in the scene.
        """
        for rb in self.robots:
            rb.stop_logging()

        for logger in self.additional_loggers:
            logger.stop_logging()

    def add_logger(self, logger):
        """
        Add additional objects implementing the Logger interface.
        Args:
            logger: Logger object
        """
        self.additional_loggers.append(logger)

    def log_data(self):
        """calls the log_data function for all loggers in the scene."""
        for rb in self.robots:
            rb.log_data()
        for logger in self.additional_loggers:
            logger.log_data()

    @property
    def step_count(self):
        return self.time_keeper.step_count

    @property
    def time_stamp(self):
        return self.time_keeper.time_stamp
