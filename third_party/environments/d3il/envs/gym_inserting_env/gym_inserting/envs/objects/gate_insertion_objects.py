import numpy as np

from d3il_sim.sims.universal_sim.PrimitiveObjects import Box, Sphere

init_end_eff_pos = [0.525, -0.28, 0.12]
# init_end_eff_pos = [0.4, -0.5, 0.12]

box_1_pos = np.array([0.4, -0.3, -0.0072])
box_1_quat = [1, 0, 0, 0]

box_2_pos = np.array([0.55, -0.3, -0.0072])
box_2_quat = [1, 0, 0, 0]

box_3_pos = np.array([0.5, -0.35, -0.0072])
box_3_quat = [1, 0, 0, 0]

target_box_1_pos = [0.3575, 0.276, 0.0]
target_box_1_quat = [0, 1, 0, 0]

target_box_2_pos = [0.525, 0.4535, 0.0]
target_box_2_quat = [0, 1, 0, 0]

target_box_3_pos = [0.6925, 0.276, 0.0]
target_box_3_quat = [0, 1, 0, 0]

def get_obj_list():
    push_box1 = Box(
        name="push_box1",
        init_pos=box_1_pos,
        init_quat=[0, 1, 0, 0],
        rgba=[1, 0, 0, 1.0],
        mass=0.05,
        size=[0.025, 0.025, 0.025],
    )

    push_box2 = Box(
        name="push_box2",
        init_pos=box_2_pos,
        init_quat=[0, 1, 0, 0],
        rgba=[0, 1, 0, 1.0],
        mass=0.05,
        size=[0.025, 0.025, 0.025],
    )

    push_box3 = Box(
        name="push_box3",
        init_pos=box_3_pos,
        init_quat=[0, 1, 0, 0],
        rgba=[0, 0, 1, 1.0],
        mass=0.05,
        size=[0.025, 0.025, 0.025],
    )

    target_box1 = Box(
        None,
        target_box_1_pos,
        target_box_1_quat,
        size=[0.025, 0.025, 0.02],
        rgba=[1, 0, 0, 0.3],
        visual_only=True,
        static=True
        # wall_height=0.005,
    )

    target_box2 = Box(
        None,
        target_box_2_pos,
        target_box_2_quat,
        size=[0.025, 0.025, 0.02],
        rgba=[0, 1, 0, 0.3],
        visual_only=True,
        static=True
        # wall_height=0.005,
    )

    target_box3 = Box(
        None,
        target_box_3_pos,
        target_box_3_quat,
        size=[0.025, 0.025, 0.02],
        rgba=[0, 0, 1, 0.3],
        visual_only=True,
        static=True
        # wall_height=0.005,
    )

    # gate
    maze_1 = Box(
        name="maze_1",
        init_pos=[0.4475, 0.15, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.037, 0.01, 0.03],
        static=True
    )

    maze_2 = Box(
        name="maze_2",
        init_pos=[0.6025, 0.15, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.037, 0.01, 0.03],
        static=True
    )

    # diag
    maze_3 = Box(
        name="maze_3",
        init_pos=[0.4, 0.17, 0.0],
        init_quat=[0, 0.5, 1, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.03, 0.01, 0.03],
        static=True
    )

    maze_4 = Box(
        name="maze_4",
        init_pos=[0.65, 0.17, 0.0],
        init_quat=[0, 0.5, -1, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.03, 0.01, 0.03],
        static=True
    )

    # long
    maze_5 = Box(
        name="maze_5",
        init_pos=[0.383, 0.2185, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.01, 0.03, 0.03],
        static=True
    )

    maze_6 = Box(
        name="maze_6",
        init_pos=[0.667, 0.2185, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.01, 0.03, 0.03],
        static=True
    )

    # goal up and down
    maze_7 = Box(
        name="maze_7",
        init_pos=[0.3525, 0.2385, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.04, 0.01, 0.03],
        static=True
    )

    maze_8 = Box(
        name="maze_8",
        init_pos=[0.6975, 0.2385, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.04, 0.01, 0.03],
        static=True
    )

    maze_9 = Box(
        name="maze_9",
        init_pos=[0.32, 0.276, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.01, 0.0475, 0.03],
        static=True
    )

    maze_10 = Box(
        name="maze_10",
        init_pos=[0.73, 0.276, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.01, 0.0475, 0.03],
        static=True
    )

    maze_11 = Box(
        name="maze_11",
        init_pos=[0.3525, 0.3135, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.04, 0.01, 0.03],
        static=True
    )

    maze_12 = Box(
        name="maze_12",
        init_pos=[0.6975, 0.3135, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.04, 0.01, 0.03],
        static=True
    )

    # long
    maze_13 = Box(
        name="maze_13",
        init_pos=[0.383, 0.3335, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.01, 0.03, 0.03],
        static=True
    )

    maze_14 = Box(
        name="maze_14",
        init_pos=[0.667, 0.3335, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.01, 0.03, 0.03],
        static=True
    )

    # diag
    maze_15 = Box(
        name="maze_15",
        init_pos=[0.435, 0.3975, 0.0],
        init_quat=[0, 0.5, 1, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.01, 0.07, 0.03],
        static=True
    )

    maze_16 = Box(
        name="maze_16",
        init_pos=[0.615, 0.3975, 0.0],
        init_quat=[0, 0.5, -1, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.01, 0.07, 0.03],
        static=True
    )

    # goal right
    maze_17 = Box(
        name="maze_17",
        init_pos=[0.4875, 0.4585, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.01, 0.04, 0.03],
        static=True
    )

    maze_18 = Box(
        name="maze_18",
        init_pos=[0.5625, 0.4585, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.01, 0.04, 0.03],
        static=True
    )

    maze_19 = Box(
        name="maze_19",
        init_pos=[0.525, 0.491, 0.0],
        init_quat=[0, 1, 0, 0],
        rgba=[0.5, 0.5, 0.5, 1.0],
        mass=0.05,
        size=[0.0475, 0.01, 0.03],
        static=True
    )

    obj_list = []

    return obj_list, push_box1, push_box2, push_box3, target_box1, target_box2, target_box3,\
        [maze_1, maze_2, maze_3, maze_4, maze_5, maze_6, maze_7, maze_8, maze_9, maze_10, maze_11,
         maze_12, maze_13, maze_14, maze_15, maze_16, maze_17, maze_18, maze_19]
