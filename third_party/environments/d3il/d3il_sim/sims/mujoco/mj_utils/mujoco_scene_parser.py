import os
import xml.etree.ElementTree as Et
from shutil import copyfile
from xml.etree.ElementTree import Element

from mujoco_py import MjSim, load_model_from_path

from environments.d3il.d3il_sim.sims.mujoco.mj_utils.mujoco_scene_object import MujocoSurrounding
from environments.d3il.d3il_sim.sims.mujoco.MujocoLoadable import MujocoIncludeTemplate, MujocoLoadable
from environments.d3il.d3il_sim.utils.sim_path import d3il_path


class MujocoSceneParser:
    """
    Class for parsing assets to the xml file for setting up the scene for a mujoco environment.
    New created assets are written at the end of the worldbody element which specifies the assets in the environment.
    """

    def __init__(self, main_xml_path=None, proc_id=""):
        if main_xml_path is None:
            main_xml_path = "./models/mujoco/surroundings/base.xml"
        src = d3il_path(main_xml_path)
        self._xml_path = d3il_path(
            "./models/mujoco/surroundings/"
        )  # Need Directory for XML-Include
        self.scene_xml = d3il_path(
            self._xml_path, "{}assembled_scene.xml".format(proc_id)
        )  # Full Path
        copyfile(src, self.scene_xml)

        # Read File
        self._tree = Et.parse(self.scene_xml)
        self._root = self._tree.getroot()
        self._worldbody = self._root.find("worldbody")
        assert self._worldbody, "Error, xml file does contain a world body."

    def create_scene(self, mj_robots: list, mj_surrounding, object_list: list, dt):

        if object_list is None:
            object_list = []
        if not isinstance(object_list, list):
            object_list = [object_list]

        self.set_surrounding(mj_surrounding)

        # adding all assets to the scene
        for obj in object_list:
            self.load_mj_loadable(obj)

        for mj_robot in mj_robots:
            self.load_mj_loadable(mj_robot)

        self.set_dt(dt)

        model = load_model_from_path(self.scene_xml)
        sim = MjSim(model=model, nsubsteps=1)

        self.cleanup(mj_robots, object_list)
        return sim, model

    def cleanup(self, mj_robots: list, object_list: list):
        os.remove(self.scene_xml)

        for obj in mj_robots + object_list:
            if isinstance(obj, MujocoIncludeTemplate):
                obj.cleanup()

    def load_mj_loadable(self, mj_loadable: MujocoLoadable):
        """
        loadas MujocoLoadable objects to the Scene XML
        Args:
            mj_loadable: a MujocoLoadable Implementation

        Returns:
            None
        """

        xml_element, include = mj_loadable.to_mj_xml(self._xml_path)
        if include:
            self._root.append(xml_element)  # append Include instruction
        else:
            self._worldbody.append(
                xml_element
            )  # append the new object to the worldbody of the xml file
        self.indent(self._root)  # ensures correct indendation

        self._tree.write(self.scene_xml)  # writes the changes to the file

    def set_dt(self, dt=0.001):
        """
        Sets 1 / number of timesteps mujoco needs for executing one second of wall-clock time by writing to the root
        xml.file.

        Args:
            dt:
                1 / number of timesteps needed for computing one second of wall-clock time

        Returns:
            No return value

        """
        options = self._root.find("option")
        options.set("timestep", str(dt))
        self.indent(self._root)  # ensures correct indentation
        self._tree.write(self.scene_xml)  # writes the changes to the file

    def indent(self, elem: Element, level=0):
        """
        Helper function:
        Ensures that the element which is written to the xml file is correctly indented.

        Returns:
            No return value
        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def set_surrounding(self, mj_surrounding):
        self.load_mj_loadable(mj_surrounding)
