import abc
import os
import xml.etree.ElementTree as Et
from typing import Tuple

from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import IncludeType


class MjLoadable(abc.ABC):
    """
    Interface for all kind of objects which can be loaded to a Mujoco Scene.
    Each class is responsible themselves to return a valid XML element which will be added to the Scene XML.
    """

    @abc.abstractmethod
    def mj_load(self) -> Tuple[Et.Element, list, IncludeType]:
        """
        Function to convert the object to a valid XML node. Needs to be implemented by the subclass.
        Returns:
            XML Element, List of XML assets, bool if it is an include element.
        """
        raise NotImplementedError


class MjXmlLoadable(MjLoadable):
    """
    Interface to include a static XML file.
    """

    def __init__(self, full_xml_path, asset_path=None) -> None:
        super().__init__()
        self._full_xml_path = full_xml_path
        self._asset_path = asset_path

    @property
    def loadable_dir(self):
        return os.path.split(self._full_xml_path)[0]

    @property
    def asset_path(self):
        """
        Returns:
           Path to the asset directory.
        """
        if self._asset_path is not None:
            return self._asset_path
        return os.path.join(self.loadable_dir, "assets")

    @property
    def file_name(self):
        return os.path.split(self._full_xml_path)[1]

    def mj_load(self) -> Tuple[Et.Element, list, IncludeType]:
        et_include = Et.Element("include")
        et_include.set("file", self.file_name)
        assets = {}

        # Assume that all referenced assets are in the assets directory
        if os.path.isdir(self.asset_path):
            for dirpath, dirnames, files in os.walk(self.asset_path):
                for f in files:
                    with open(os.path.join(dirpath, f), "rb") as file:
                        assets[f] = file.read()

        # Load one self
        with open(os.path.join(self._full_xml_path), "rb") as file:
            assets[self.file_name] = file.read()
        return et_include, assets, IncludeType.FILE_INCLUDE


class MjIncludeTemplate(MjXmlLoadable):
    """
    Interface to include a template XML file.
    """

    def __init__(self, full_xml_path, asset_path=None) -> None:
        super().__init__(full_xml_path, asset_path)
        self._tmp_filled_xml = None

    @abc.abstractmethod
    def modify_template(self, et: Et.ElementTree) -> Et.ElementTree:
        """read the include template. Modify it and save a copy. Return copy path

        Args:
            et (Et.ElementTree): parsed XML include template

        Returns:
            Et.ElementTree: modified element tree element
        """

    def mj_load(self) -> Tuple[Et.Element, list, IncludeType]:
        inc, assets, include_type = super().mj_load()

        obj = Et.parse(self._full_xml_path)
        new_xml = self.modify_template(obj)

        self._tmp_filled_xml = new_xml

        et_include = Et.Element("include")
        et_include.set("file", new_xml)
        return et_include, assets, include_type

    def cleanup(self):
        if self._tmp_filled_xml is not None:
            os.remove(self._tmp_filled_xml)


class MjFreezable:
    @abc.abstractmethod
    def freeze(self, data, model):
        """
        Freeze the object. This should override the initial position and
        orientation of the object with the current position and orientation.
        These new values are then used upon reload of the scene.
        """
        pass

    @abc.abstractmethod
    def unfreeze(self, data, model):
        """
        Unfreeze is performed after reconstruction of scene and is called after
        freeze command. This function can be used to reset velocities, inertia etc.
        of the freezed object, since these attributes are lost after reconstruction.
        """
        pass
