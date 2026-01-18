import os

import numpy as np
import xml.etree.ElementTree as Et
from typing import Tuple

from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import IncludeType
from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box, Sphere
from environments.d3il.d3il_sim.core.sim_object.sim_object import SimObject
from environments.d3il.d3il_sim.sims.mj_beta.MjLoadable import MjXmlLoadable
from environments.d3il.d3il_sim.utils import sim_path

init_end_eff_pos = [0.525, -0.35, 0.25]

box_pos = np.array([0.6, 0.15, 0.0])
box_quat = [1, 0, 0, 0]


class PushObject(SimObject, MjXmlLoadable):
    def __init__(self, file_name, object_name, pos, quat, root=sim_path.D3IL_DIR):
        if pos is None:
            pos = [0, 0, 0]
        else:
            assert len(pos) == 3, "Error, parameter pos has to be three dimensional."

        if quat is None:
            quat = [0, 0, 0, 0]
        else:
            assert len(quat) == 4, "Error, parameter quat has to be four dimensional."

        self.obj_dir_path = "./models/mj/common-objects/robot_push_box/" + file_name
        self.root = root
        self.pos = pos
        self.quat = quat
        self.name = object_name

        SimObject.__init__(self, name=self.name, init_pos=self.pos, init_quat=self.quat)
        MjXmlLoadable.__init__(self, os.path.join(root, self.obj_dir_path))

    def get_poi(self) -> list:
        """

        Returns:
            a list of points of interest for the scene to query
        """
        return [self.name]


def get_obj_list():

    push_box = PushObject(
        file_name="robot_push_box.xml",
        object_name="aligning_box",
        pos=box_pos,
        quat=box_quat
    )

    target_box = PushObject(
        file_name="target_box.xml",
        object_name="target_box",
        pos=box_pos,
        quat=box_quat
    )

    obj_list = [push_box, target_box]

    return obj_list

    obj_list += [
        # TARGET
        Sphere(None, [0.35, 0.20, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.35, 0.40, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.7, 0.20, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.7, 0.40, 0], [0, 1, 0, 0], static=True, visual_only=True),
        # WORKSPACE
        # Sphere(
        #     None,
        #     [0.30, -0.45, 0],
        #     [0, 1, 0, 0],
        #     static=True,
        #     visual_only=True,
        #     rgba=[0, 1, 0, 1],
        # ),
        # Sphere(
        #     None,
        #     [0.30, 0.45, 0],
        #     [0, 1, 0, 0],
        #     static=True,
        #     visual_only=True,
        #     rgba=[0, 1, 0, 1],
        # ),
        # Sphere(
        #     None,
        #     [0.8, -0.45, 0],
        #     [0, 1, 0, 0],
        #     static=True,
        #     visual_only=True,
        #     rgba=[0, 1, 0, 1],
        # ),
        # Sphere(
        #     None,
        #     [0.8, 0.45, 0],
        #     [0, 1, 0, 0],
        #     static=True,
        #     visual_only=True,
        #     rgba=[0, 1, 0, 1],
        # ),
        # START
        Sphere(
            None,
            [0.35, -0.35, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.35, -0.1, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.7, -0.35, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.7, -0.1, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
    ]

    return obj_list
