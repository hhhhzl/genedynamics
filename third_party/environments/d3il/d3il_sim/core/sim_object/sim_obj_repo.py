from typing import List

from environments.d3il.d3il_sim.utils.unique_dict import UniqueDict

from .sim_object import SimObject


class SimObjectRepository:
    """
    Repository class controlling access to the SimObjects in a Scene.
    """

    def __init__(self, obj_list=None) -> None:
        # dict mapping names to obj_ids
        self._name2id_map = UniqueDict(err_msg="Duplicate object name:")
        # dict mapping obj_ids to name
        self._id2name_map = UniqueDict(err_msg="Duplicate object id:")
        # dict mapping names to object instances
        self._objects = UniqueDict(err_msg="Duplicate object name:")

        if obj_list is not None:
            for obj in obj_list:
                self.add_object(obj)

    def add_object(self, sim_obj: SimObject) -> None:
        self._objects[sim_obj.name] = sim_obj

    def remove_object(self, sim_obj: SimObject) -> None:
        # Delete name references
        del self._name2id_map[sim_obj.name]
        # del self._id2name_map[sim_obj.obj_id]

        # Delete Instance from map
        del self._objects[sim_obj.name]

    def register_obj_id(self, sim_obj: SimObject, obj_id: int):
        sim_obj.obj_id = obj_id
        self._name2id_map[sim_obj.name] = obj_id

    def get_obj_list(self) -> List[SimObject]:
        return list(self._objects.values())

    def get_object(self, name: str = None, obj_id: int = None) -> SimObject:
        """getter to retrieve a SimObject from the repository.

        Args:
            name (str, optional): the SimObject's name. Defaults to None.
            obj_id (int, optional): the SimObject's id. Defaults to None.

        Returns:
            SimObject
        """
        if obj_id is not None:
            name = self.get_name_from_id(obj_id)
        return self._objects[name]

    def get_id_from_name(self, name: str) -> int:
        """getter for the obj_id belonging to the object with this name.

        Args:
            name (str): object name

        Returns:
            int: object id
        """
        return self._name2id_map[name]

    def get_name_from_id(self, obj_id: int) -> str:
        """getter for the name belonging to the object with this obj_id

        Args:
            obj_id (int): an object id

        Returns:
            str: the object's name
        """
        return self._id2name_map[obj_id]
