import logging
import os

import pybullet as p

from environments.d3il.d3il_sim.core.sim_object.sim_object import SimObject
from environments.d3il.d3il_sim.sims.pybullet.PybulletLoadable import PybulletLoadable
from environments.d3il.d3il_sim.utils.geometric_transformation import euler2quat, wxyz_to_xyzw
from environments.d3il.d3il_sim.utils.sim_path import d3il_path


class PyBulletURDFObject(SimObject, PybulletLoadable):
    def __init__(self, urdf_name, object_name, position, orientation, data_dir=None):
        """Creates a Legacy URDF Object

        Args:
            urdf_name (str): URDF file name
            object_name (str): object name
            position: initial position
            orientation: initial orientation as euler angle
            data_dir (optional): direcotry of URDF files. Defaults to None.

        Raises:
            ValueError: Error if URDF file cannot be found
        """
        # euler to quat:
        init_quat = euler2quat(orientation)
        super().__init__(object_name, position, init_quat)

        if data_dir is None:
            data_dir = d3il_path("./models/pybullet/objects/")

        objects = os.listdir(data_dir)

        if not urdf_name.endswith(".urdf"):
            if urdf_name + ".urdf" in objects:
                urdf_name += ".urdf"
            else:
                raise ValueError(
                    "Error, object with name "
                    + urdf_name
                    + " not found. Check that a object with the "
                    "specified name exists in the data "
                    "directory"
                )

        assert len(position) == 3, "Error, <position> has three entries x, y, z."
        assert (
            len(orientation) == 3
        ), "Error, <orientation> has three entries yaw, pitch, and roll."

        self.__urdf_name = urdf_name
        self.__object_name = object_name
        self.__position = position
        self.__orientation = orientation
        self.__data_dir = data_dir

    @property
    def urdf_name(self):
        return self.__urdf_name

    @property
    def object_name(self):
        return self.__object_name

    @property
    def position(self):
        return self.__position

    @property
    def orientation(self):
        return self.__orientation

    @property
    def data_dir(self):
        return self.__data_dir

    def get_poi(self) -> list:
        return [self.object_name]

    def pb_load(self, pb_sim):
        path_to_urdf = self.data_dir + "/" + self.urdf_name
        orientation = self.orientation
        position = self.position
        fixed = 0
        inertia_from_file = False

        obj_urdf = path_to_urdf
        orientation = list(orientation)
        if len(orientation) == 3:
            orientation = p.getQuaternionFromEuler(orientation)
        position = list(position)

        try:
            if inertia_from_file is True:
                id = p.loadURDF(
                    obj_urdf,
                    position,
                    orientation,
                    fixed,
                    flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
                    physicsClientId=pb_sim,
                )
            else:
                id = p.loadURDF(
                    obj_urdf,
                    position,
                    orientation,
                    fixed,
                    flags=p.URDF_USE_SELF_COLLISION,
                    physicsClientId=pb_sim,
                )

        except Exception:
            logging.getLogger(__name__).error("Stopping the program")
            raise ValueError(
                "Could not load URDF-file: Check the path to file. Stopping the program."
                "Your path:",
                obj_urdf,
            )
        return id
