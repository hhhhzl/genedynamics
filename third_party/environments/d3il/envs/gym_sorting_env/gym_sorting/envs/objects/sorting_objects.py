import numpy as np
import os
import xml.etree.ElementTree as Et
from typing import Tuple

from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box, Sphere, Box
from environments.d3il.d3il_sim.core.sim_object.sim_object import SimObject
from environments.d3il.d3il_sim.sims.mj_beta.MjLoadable import MjXmlLoadable
from environments.d3il.d3il_sim.utils import sim_path

init_end_eff_pos = [0.525, -0.3, 0.25]
#init_end_eff_pos = [0.4, -0.5, 0.12]
box_pos1 = np.array([0.5, -0.1, 0.0])

platform_pos = np.array([0.5, -0.1, 0.0])
platform_quat = [1, 0, 0, 0]


class SortingObject(SimObject, MjXmlLoadable):
    def __init__(self, file_name, object_name, pos, quat, root=sim_path.D3IL_DIR):
        if pos is None:
            pos = [0, 0, 0]
        else:
            assert len(pos) == 3, "Error, parameter pos has to be three dimensional."

        if quat is None:
            quat = [0, 0, 0, 0]
        else:
            assert len(quat) == 4, "Error, parameter quat has to be four dimensional."

        self.obj_dir_path = "./models/mj/common-objects/sorting/" + file_name
        self.root = root
        self.pos = pos
        self.quat = quat
        self.name = object_name

        SimObject.__init__(self, name=self.name, init_pos=self.pos, init_quat=self.quat)
        MjXmlLoadable.__init__(self, os.path.join(root, self.obj_dir_path))

    def mj_load(self) -> Tuple[Et.Element, list, bool]:
        include_et, xml, is_include = super(SortingObject, self).mj_load()
        xml_file = list(xml.values())[0]  # in this case we only expect one file
        obj = Et.ElementTree(Et.fromstring(xml_file))
        worldbody = obj.find("worldbody")
        body = worldbody.find("body")

        # cast types to string for xml parsing
        obj_pos_str = " ".join(map(str, self.pos))
        obj_quat_str = " ".join(map(str, self.quat))

        body.set("pos", obj_pos_str)
        body.set("quat", obj_quat_str)

        return (
            include_et,
            {
                os.path.join(self.loadable_dir, self.file_name): Et.tostring(
                    obj.getroot()
                )
            },
            is_include,
        )

    def get_poi(self) -> list:
        """

        Returns:
            a list of points of interest for the scene to query
        """
        return [self.name]


def get_obj_list():

    platform = SortingObject(
        file_name="platform.xml",
        object_name="platform",
        pos=platform_pos,
        quat=platform_quat
    )

    red_box_1 = Box(
        name="red_1",
        init_pos=box_pos1,
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1.0],
        mass=0.05,
        size=[0.03, 0.03, 0.03],
    )

    red_box_2 = Box(
        name="red_2",
        init_pos=box_pos1,
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1.0],
        mass=0.05,
        size=[0.03, 0.03, 0.03],
    )

    red_box_3 = Box(
        name="red_3",
        init_pos=box_pos1,
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1.0],
        mass=0.05,
        size=[0.03, 0.03, 0.03],
    )

    blue_box_1 = Box(
        name="blue_1",
        init_pos=box_pos1,
        init_quat=[0, 1, 0, 0],
        rgba=[0, 0, 1, 1.0],
        mass=0.05,
        size=[0.03, 0.03, 0.03],
    )

    blue_box_2 = Box(
        name="blue_2",
        init_pos=box_pos1,
        init_quat=[0, 1, 0, 0],
        rgba=[0, 0, 1, 1.0],
        mass=0.05,
        size=[0.03, 0.03, 0.03],
    )

    blue_box_3 = Box(
        name="blue_3",
        init_pos=box_pos1,
        init_quat=[0, 1, 0, 0],
        rgba=[0, 0, 1, 1.0],
        mass=0.05,
        size=[0.03, 0.03, 0.03],
    )

    target_box_1 = Box(
        name="target_box_1",
        init_pos=[0.4, 0.41, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 0.5],
        mass=0.05,
        size=[0.1, 0.01, 0.1],
        static=True
    )

    target_box_2 = Box(
        name="target_box_2",
        init_pos=[0.3, 0.32, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 0.5],
        mass=0.05,
        size=[0.005, 0.1, 0.1],
        static=True
    )

    target_box_3 = Box(
        name="target_box_3",
        init_pos=[0.5, 0.32, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 0.5],
        mass=0.05,
        size=[0.005, 0.1, 0.1],
        static=True
    )

    target_box_4 = Box(
        name="target_box_4",
        init_pos=[0.4, 0.22, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 0.5],
        mass=0.05,
        size=[0.1, 0.005, 0.1],
        static=True
    )

    target_box_5 = Box(
        name="target_box_5",
        init_pos=[0.625, 0.41, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0, 0, 1, 0.5],
        mass=0.05,
        size=[0.1, 0.01, 0.1],
        static=True
    )

    target_box_6 = Box(
        name="target_box_6",
        init_pos=[0.525, 0.32, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0, 0, 1, 0.5],
        mass=0.05,
        size=[0.005, 0.1, 0.1],
        static=True
    )

    target_box_7 = Box(
        name="target_box_7",
        init_pos=[0.725, 0.32, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0, 0, 1, 0.5],
        mass=0.05,
        size=[0.005, 0.1, 0.1],
        static=True
    )

    target_box_8 = Box(
        name="target_box_8",
        init_pos=[0.625, 0.22, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0, 0, 1, 0.5],
        mass=0.05,
        size=[0.1, 0.005, 0.1],
        static=True
    )

    red_boxes = [red_box_1, red_box_2, red_box_3]
    blue_boxes = [blue_box_1, blue_box_2, blue_box_3]

    obj_list = [target_box_1, target_box_2, target_box_3, target_box_4, target_box_5, target_box_6, target_box_7, target_box_8,
                platform]

    return red_boxes, blue_boxes, obj_list
