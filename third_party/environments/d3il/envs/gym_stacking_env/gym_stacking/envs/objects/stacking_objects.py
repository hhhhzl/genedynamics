import numpy as np
import os
import xml.etree.ElementTree as Et
from typing import Tuple

from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box, Sphere, Box
from environments.d3il.d3il_sim.core.sim_object.sim_object import SimObject
from environments.d3il.d3il_sim.sims.mj_beta.MjLoadable import MjXmlLoadable
from environments.d3il.d3il_sim.utils import sim_path

init_end_eff_pos = [0.525, 0.0, 0.3]

box_pos1 = np.array([0.5, -0.1, 0.0])
box_pos2 = np.array([0.5, 0, 0.0])
box_pos3 = np.array([0.5, 0.1, 0.0])

target_pos = [0.5, 0.2, 0]
target_quat = [0, 1, 0, 0]


def get_obj_list():

    red_box = Box(
        name="red_box",
        init_pos=box_pos1,
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1.0],
        mass=0.05,
        size=[0.03, 0.03, 0.03],
        # visual_only=True,
    )

    green_box = Box(
        name="green_box",
        init_pos=box_pos2,
        init_quat=[0, 1, 0, 0],
        rgba=[0, 1, 0, 1.0],
        mass=0.05,
        size=[0.03, 0.03, 0.03],
        # visual_only=True,
    )

    blue_box = Box(
        name="blue_box",
        init_pos=box_pos2,
        init_quat=[0, 1, 0, 0],
        rgba=[0, 0, 1, 1.0],
        mass=0.05,
        size=[0.03, 0.05, 0.03],
        # visual_only=True,
    )

    target_box = Box(
        name="target_box",
        init_pos=target_pos,
        init_quat=target_quat,
        size=[0.05, 0.05, 0.04],
        rgba=[1, 0.65, 0, 0.3],
        visual_only=True,
        static=True
        # wall_height=0.005,
    )

    obj_list = [red_box, green_box, blue_box, target_box]

    return obj_list

    obj_list += [
        # TARGET
        Sphere(None, [0.35, 0.1, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.35, 0.3, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.65, 0.1, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.65, 0.3, 0], [0, 1, 0, 0], static=True, visual_only=True),
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
            [0.35, -0.3, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.35, 0.05, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.65, -0.3, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.65, 0.05, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
    ]

    return obj_list
