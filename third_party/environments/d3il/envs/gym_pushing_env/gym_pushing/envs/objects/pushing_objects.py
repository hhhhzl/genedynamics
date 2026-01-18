import numpy as np

from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box, Sphere

init_end_eff_pos = [0.525, -0.28, 0.12]
# init_end_eff_pos = [0.4, -0.5, 0.12]

box_pos1 = np.array([0.4, -0.3, -0.0072])
box_pos2 = np.array([0.5, -0.3, -0.0072])

target_pos1 = [0.42, +0.3, 0]
target_quat1 = [0, 1, 0, 0]

target_pos2 = [0.63, 0.3, 0]
target_quat2 = [0, 1, 0, 0]


def get_obj_list():
    push_box1 = Box(
        name="push_box",
        init_pos=box_pos1,
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1.0],
        mass=0.05,
        size=[0.03, 0.03, 0.03],
        # visual_only=True,
    )

    push_box2 = Box(
        name="push_box2",
        init_pos=box_pos2,
        init_quat=[0, 1, 0, 0],
        rgba=[0, 1, 0, 1.0],
        mass=0.05,
        size=[0.03, 0.03, 0.03],
        # visual_only=True,
    )

    target_box_1 = Box(
        None,
        target_pos1,
        target_quat1,
        size=[0.05, 0.05, 0.04],
        rgba=[1, 0, 0, 0.3],
        visual_only=True,
        static=True
        # wall_height=0.005,
    )

    target_box_2 = Box(
        None,
        target_pos2,
        target_quat2,
        size=[0.05, 0.05, 0.04],
        rgba=[0, 1, 0, 0.3],
        visual_only=True,
        static=True
        # wall_height=0.005,
    )

    # obj_list = [push_box1, push_box2, target_box_1, target_box_2]

    obj_list = []

    return obj_list, push_box1, push_box2, target_box_1, target_box_2

    obj_list += [
        # TARGET
        Sphere(None, [0.35, 0.20, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.35, 0.40, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.7, 0.20, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.7, 0.40, 0], [0, 1, 0, 0], static=True, visual_only=True),
        # WORKSPACE
        Sphere(
            None,
            [0.30, -0.45, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 0, 1],
        ),
        Sphere(
            None,
            [0.30, 0.45, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 0, 1],
        ),
        Sphere(
            None,
            [0.8, -0.45, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 0, 1],
        ),
        Sphere(
            None,
            [0.8, 0.45, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 0, 1],
        ),
        # START
        Sphere(
            None,
            [0.35, -0.18, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.35, 0.06, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.7, -0.18, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.7, 0.06, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
    ]

    return obj_list, push_box1, push_box2, target_box_1, target_box_2
