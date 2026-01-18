import abc
from functools import wraps

from environments.d3il.d3il_sim.core import Camera, RobotBase, Scene
from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import PrimitiveObject


def _register_function_on_class(cls, fn):
    """registers a function fn on a class cls

    Args:
        fn (function): a function
    """

    @wraps(fn)
    def self_wrapper(self, *args, **kwargs):
        return fn(self, *args, **kwargs)

    setattr(cls, fn.__name__, self_wrapper)


class SimFactory(abc.ABC):
    """Abstract Base Class for a Factory-Like construct for a Simulator.
    Each Simulator should implement this Factory, to provide access to its Simulator specific interface implementations.
    By using the factories, robot experiments can be defined without much knowledge of the simulator internals
    """

    def __init__(self) -> None:
        super().__init__()
        fn = self.prim_loading()
        if fn is not None:
            _register_function_on_class(PrimitiveObject, fn)

    @abc.abstractmethod
    def create_scene(
        self,
        gin_config=None,
        object_list: list = None,
        dt: float = 0.001,
        render: Scene.RenderMode = Scene.RenderMode.HUMAN,
        *args,
        **kwargs
    ) -> Scene:
        """create a Scene for this Simulator

        Args:
            robot (RobotBase): a matching robot
            object_list (list, optional): a list of SimObjects to load into the scene. Defaults to None.
            dt (float, optional): the minimal timestep for the simulation. Defaults to 0.001.
            render (Scene.RenderMode, optional): RenderMode for the Scene. Defaults to Scene.RenderMode.HUMAN.

        Returns:
            Scene: a Scene object
        """
        pass

    @abc.abstractmethod
    def create_robot(self, scene, *args, **kwargs) -> RobotBase:
        """
        create a Robot for this Simulator
        Returns:
            RobotBase: a Robot
        """
        pass

    @abc.abstractmethod
    def create_camera(
        self,
        name: str,
        width: int = 1000,
        height: int = 1000,
        init_pos=None,
        init_quat=None,
        *args,
        **kwargs
    ) -> Camera.Camera:
        """create a Camera for this Simulator

        Args:
            name (str): camera name
            width (int): camera image width
            height (int): camera image height
            init_pos (vec3, optional): XYZ Position at which the camera spawns. Defaults to None.
            init_quat (vec4, optional): WXYZ Orientation at which the camera spawns. Defaults to None.
        Returns:
            Camera: a Camera
        """
        pass

    @abc.abstractmethod
    def prim_loading(self):
        """return a PrimitiveObject Loading function
        Returns:
            fn: loading function
        """
        return None

    @property
    def RenderMode(self):
        """
        Choose Scene RenderMode
        Returns:
            RenderMode enum for Scene
        """
        return Scene.RenderMode


class SimRepository:
    """a static repository, in which all available SimFactories are registered"""

    _repository = {}

    @classmethod
    def register(cls, factory: SimFactory, sim_name: str):
        """registers a SimFactory for easy use

        Args:
            factory (SimFactory): the SimFactory Object
            sim_name (str): the name under which the SimFactory can be accessed.
        """
        if sim_name in cls._repository:
            return
            # raise KeyError("The sim_name {} is already in use.".format(sim_name))
        cls._repository[sim_name] = factory

    @classmethod
    def get_factory(cls, sim_name: str) -> SimFactory:
        """returns the SimFactory belonging to the named Simulator

        Args:
            sim_name (str): a simulator name

        Returns:
            SimFactory: the simulator's SimFactory Object
        """
        return cls._repository[sim_name]

    @classmethod
    def list_all_sims(cls):
        """lists all available Simulators which have been registered at the repository.

        Returns:
            a list of simulator names
        """
        return cls._repository.keys()
