from typing import Tuple
from xml.etree import ElementTree as Et

from environments.d3il.d3il_sim.core.sim_object.sim_object import SimObject
from environments.d3il.d3il_sim.sims.mujoco.MujocoLoadable import MujocoLoadable


class VirtualGraspObject(SimObject, MujocoLoadable):
    def __init__(
        self,
        init_pos,
        init_quat,
        rgba=None,
        scale: float = 1.0,
        visual_only: bool = False,
    ):
        super().__init__(None, init_pos, init_quat)

        if rgba is None:
            rgba = [0.5, 0.5, 0.5, 1]

        self.rgba = rgba

        self.scale = scale
        self.visual_only = visual_only
        self.mass = 0.1

    def get_poi(self) -> list:
        return [self.name]

    def stringify_scale(self, attribute_list):
        scaled_attributes = [str(x * self.scale) for x in attribute_list]
        return " ".join(scaled_attributes)

    def to_mj_xml(self, scene_dir: str) -> Tuple[Et.Element, bool]:
        grip_size = [0.06, 0.02, 0.03]
        foot_size = [0.06, 0.06, 0.01]
        foot_offset = [0.0, 0.0, -grip_size[2] / 2.0 - foot_size[2] / 2.0]

        pos_str = " ".join(map(str, self.init_pos))
        orientation_str = " ".join(map(str, self.init_quat))
        mass_str = str(self.mass)
        rgba_str = " ".join(map(str, self.rgba))

        # Create new object for xml tree
        object_body = Et.Element("body")
        object_body.set("name", self.name)
        object_body.set("pos", pos_str)
        object_body.set("quat", orientation_str)

        grip = Et.SubElement(object_body, "geom")
        grip.set("type", "box")
        # if self.mass:
        #    grip.set('mass', mass_str)
        grip.set("size", self.stringify_scale(grip_size))
        grip.set("rgba", rgba_str)

        foot = Et.SubElement(object_body, "geom")
        foot.set("type", "box")
        foot.set("size", self.stringify_scale(foot_size))
        foot.set("pos", self.stringify_scale(foot_offset))
        foot.set("rgba", rgba_str)

        if self.visual_only:
            grip.set("contype", "0")
            grip.set("conaffinity", "0")
            foot.set("contype", "0")
            foot.set("conaffinity", "0")
        else:
            Et.SubElement(object_body, "freejoint")

        return object_body, False
