import abc
from typing import List


class SimObject(abc.ABC):
    """Abstract Base Class for Simulation Objects.
    Usually all Objects that can be loaded into a Scene should be a SimObject.
    The Scene can use the SimObject interface to query its position and orientation data from the Simulation.
    """

    GLOBAL_NAME_COUNTER = 0

    def __init__(self, name: str = None, init_pos=None, init_quat=None):
        """

        Args:
            name (str, optional): name for the object, has to be unique. If unspecified a unique name will be auto-generated.
                                    Defaults to None.
            init_pos (vec3, optional): XYZ position at which the object is initialized. Defaults to None.
            init_quat (vec4, optional): WXYZ quaternion at which the object is initialized. Defaults to None.
        """
        if name is None:
            name = "SIM_OBJ_{}".format(SimObject.GLOBAL_NAME_COUNTER)

        self.name = name

        # Initial Pos and Quat. Can be out of date and should NOT be used for computations during simulation.
        self.init_pos = init_pos
        self.init_quat = init_quat

        # Unqiue Object ID, set by Scene after loading the object
        self.obj_id = None

        # Increase Static counter for auto-generating names
        SimObject.GLOBAL_NAME_COUNTER += 1

    @abc.abstractmethod
    def get_poi(self) -> list:
        """

        Returns:
            a list of points of interest for the scene to query
        """
        return [self.name]


class IntelligentSimObject(SimObject, abc.ABC):
    """Abstract Base Class for "intelligent" simulation objects.
    Such objects might need access to the internal simulation state.
    """

    def __init__(self, name: str = None, init_pos=None, init_quat=None):
        """

        Args:
            name (str, optional): name for the object, has to be unique. If unspecified a unique name will be auto-generated.
                                    Defaults to None.
            init_pos (vec3, optional): XYZ position at which the object is initialized. Defaults to None.
            init_quat (vec4, optional): WXYZ quaternion at which the object is initialized. Defaults to None.
        """
        super(IntelligentSimObject, self).__init__(name, init_pos, init_quat)

        # Simulator Information
        self.sim = None
        self.sim_name = None

    def register_sim(self, sim, sim_name):
        """registers a simulation with the object.

        Args:
            sim: a simulator
            sim_name (str): the simulator name. can be used by the IntelligentSimObject to choose a strategy depending on the simulator.
        """
        self.sim = sim
        self.sim_name = sim_name


class DummyObject(SimObject):
    def __init__(self, pois: List[str]):
        self.pois = pois

    def get_poi(self) -> list:
        return self.pois
