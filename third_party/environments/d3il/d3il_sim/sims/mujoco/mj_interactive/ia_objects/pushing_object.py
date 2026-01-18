from typing import Tuple
from xml.etree import ElementTree as Et

from environments.d3il.d3il_sim.core.sim_object.sim_object import SimObject
from environments.d3il.d3il_sim.sims.mujoco.MujocoLoadable import MujocoLoadable


class VirtualPushObject(SimObject, MujocoLoadable):
    def __init__(
        self,
        init_pos,
        init_quat,
        rgba=None,
        scale: float = 1.0,
        visual_only: bool = False,
        name: str = None,
    ):
        super().__init__(name, init_pos, init_quat)

        if rgba is None:
            rgba = [0.5, 0.5, 0.5, 1]

        self.rgba = rgba

        self.scale = scale
        self.visual_only = visual_only
        self.mass = 1

    def get_poi(self) -> list:
        return [self.name]

    def stringify_scale(self, attribute_list):
        scaled_attributes = [str(x * self.scale) for x in attribute_list]
        return " ".join(scaled_attributes)

    def to_mj_xml(self, scene_dir: str) -> Tuple[Et.Element, bool]:
        base_size = [0.05, 0.05, 0.01]
        wall_size = [
            [0.002, base_size[1], 0.045],
            [base_size[0], 0.002, 0.045],
            [0.002, base_size[1], 0.045],
            [base_size[0], 0.002, 0.045],
        ]
        wall_offset_list = [
            [base_size[0], 0, wall_size[0][2] / 2.0 + 0.026],
            [0, base_size[1], wall_size[1][2] / 2.0 + 0.026],
            [-base_size[0], 0, wall_size[2][2] / 2.0 + 0.026],
            [0, -base_size[1], wall_size[3][2] / 2.0 + 0.026],
        ]

        pos_str = " ".join(map(str, self.init_pos))
        orientation_str = " ".join(map(str, self.init_quat))
        mass_str = str(self.mass)
        rgba_str = " ".join(map(str, self.rgba))

        # Create new object for xml tree
        object_body = Et.Element("body")
        object_body.set("name", self.name)
        object_body.set("pos", pos_str)
        object_body.set("quat", orientation_str)

        base = Et.SubElement(object_body, "geom")
        base.set("type", "box")
        base.set("mass", mass_str)
        base.set("size", self.stringify_scale(base_size))
        base.set("rgba", rgba_str)
        if self.visual_only:
            base.set("contype", "2")
            base.set("conaffinity", "2")
        base.set("friction", "0.3 0.001 0.0001")
        base.set("priority", "1")

        for i in range(4):
            wall = Et.SubElement(object_body, "geom")
            wall.set("type", "box")
            wall.set("size", self.stringify_scale(wall_size[i]))
            wall.set("pos", self.stringify_scale(wall_offset_list[i]))
            # wall.set('euler', ' '.join([str(x) for x in [0, 0, i * math.pi / 2.0]]))
            wall.set("rgba", rgba_str)
            wall.set("mass", str(0.001))
            if self.visual_only:
                wall.set("contype", "0")
                wall.set("conaffinity", "0")

        indicator = Et.SubElement(object_body, "geom")
        indicator.set("type", "box")
        indicator.set("size", self.stringify_scale([0.01] * 3))
        indicator.set(
            "pos",
            self.stringify_scale(
                [0.12, 0, 0.04],
            ),
        )
        indicator.set("rgba", rgba_str)
        indicator.set("mass", str(0.0))
        indicator.set("contype", "0")
        indicator.set("conaffinity", "0")

        if not self.visual_only:
            Et.SubElement(object_body, "freejoint")
            # j = Et.SubElement(object_body, "joint")
            # j.set("type", "slide")
            # j.set("axis", "1 0 0")

            # j = Et.SubElement(object_body, "joint")
            # j.set("type", "slide")
            # j.set("axis", "0 1 0")

            # j = Et.SubElement(object_body, "joint")
            # j.set("type", "slide")
            # j.set("axis", "0 0 1")

            # j = Et.SubElement(object_body, "joint")
            # j.set("type", "hinge")
            # j.set("axis", "0 0 1")

        return object_body, False
