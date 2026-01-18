import abc
import os
import xml.etree.ElementTree as Et
from typing import Tuple


class MujocoLoadable(abc.ABC):
    """
    Interface for all kind of objects which can be loaded to a Mujoco Scene.
    Each class is responsible themselves to return a valid XML element which will be added to the Scene XML.
    """

    @abc.abstractmethod
    def to_mj_xml(self, scene_dir: str) -> Tuple[Et.Element, bool]:
        """
        Function to convert the object to a valid XML node. Needs to be implemented by the subclass.
        Args:
            scene_dir: Directory of the scene XML. Used to built relative paths for <include /> nodes

        Returns:
            XML Element, bool if it is an include element.
        """
        raise NotImplementedError


class MujocoXmlLoadable(MujocoLoadable):
    """
    Interface to include a static XML file.
    """

    @property
    @abc.abstractmethod
    def xml_file_path(self):
        """
        Returns:
           path to the static xml file.
        """
        pass

    def to_mj_xml(self, scene_dir: str) -> Tuple[Et.Element, bool]:
        include = Et.Element("include")
        include.set("file", os.path.relpath(self.xml_file_path, scene_dir))
        return include, True


class MujocoIncludeTemplate(MujocoLoadable):
    """
    Interface to include a template XML file.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tmp_filled_xml = None

    @property
    @abc.abstractmethod
    def xml_file_path(self):
        """
        Returns:
           path to the template include xml file.
        """
        pass

    @abc.abstractmethod
    def modify_template(self, et: Et.ElementTree) -> str:
        """read the include template. Modify it and save a copy. Return copy path

        Args:
            et (Et.ElementTree): parsed XML include template

        Returns:
            str: path to the new modified copy
        """

    def to_mj_xml(self, scene_dir: str) -> Tuple[Et.Element, bool]:
        obj = Et.parse(self.xml_file_path)

        new_xml = self.modify_template(obj)

        self._tmp_filled_xml = new_xml

        include = Et.Element("include")
        include.set("file", os.path.relpath(new_xml, scene_dir))
        return include, True

    def cleanup(self):
        if self._tmp_filled_xml is not None:
            os.remove(self._tmp_filled_xml)


class MujocoTemplateObject(MujocoLoadable):
    """
    Interface to parse an XML Object definition, modify it, and make it loadable.
    """

    @property
    @abc.abstractmethod
    def xml_file_path(self):
        pass

    @abc.abstractmethod
    def fill_template(self, body: Et.Element) -> Et.Element:
        """
        Make any adjustments as needed on the XML file content.
        Probably modify at least the names and IDs of unique elements in the template
        Args:
            body:
                The parsed XML Mujoco <body> tag.
        Returns:
            a modified <body> tag
        """
        pass

    def to_mj_xml(self, scene_dir: str) -> Tuple[Et.Element, bool]:
        obj = Et.parse(self.xml_file_path)
        worldbody = obj.find("worldbody")
        body = worldbody.find("body")
        body = self.fill_template(body)
        return body, False
