import numpy as np

from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box, Sphere, Cylinder

init_end_eff_pos = [0.525, -0.28, 0.12]


def get_obj_list():
    mid_pos = 0.5
    offset = 0.075
    first_level_y = -0.1
    level_distance = 0.18
    obj_list = [
        Cylinder(name='l1_obs',
                 init_pos=[mid_pos, first_level_y, 0],
                 init_quat=[1, 0, 0, 0],
                 size=[0.03, 0.07],
                 rgba=[1, 0, 0, 1],
                 static=True),

        Cylinder(name='l2_top_obs',
                 init_pos=[mid_pos - offset, first_level_y + level_distance, 0],
                 init_quat=[1, 0, 0, 0],
                 size=[0.025, 0.1],
                 rgba=[1, 0, 0, 1],
                 static=True),

        Cylinder(name='l2_bottom_obs',
                 init_pos=[mid_pos + offset, first_level_y + level_distance, 0],
                 init_quat=[1, 0, 0, 0],
                 size=[0.025, 0.1],
                 rgba=[1, 0, 0, 1],
                 static=True),

        Cylinder(name='l3_top_obs',
                 init_pos=[mid_pos - 2 * offset, first_level_y + 2 * level_distance, 0],
                 init_quat=[1, 0, 0, 0],
                 size=[0.025, 0.1],
                 rgba=[1, 0, 0, 1],
                 static=True),

        Cylinder(name='l3_mid_obs',
                 init_pos=[mid_pos, first_level_y + 2 * level_distance, 0],
                 init_quat=[1, 0, 0, 0],
                 size=[0.025, 0.1],
                 rgba=[1, 0, 0, 1],
                 static=True),

        Cylinder(name='l3_bottom_obs',
                 init_pos=[mid_pos + 2 * offset, first_level_y + 2 * level_distance, 0],
                 init_quat=[1, 0, 0, 0],
                 size=[0.025, 0.1],
                 rgba=[1, 0, 0, 1],
                 static=True),

        Box(name='finish_line',
            init_pos=[0.4, first_level_y + 2.5 * level_distance, 0],
            init_quat=[1, 0, 0, 0],
            size=[0.5, 0.01, 0.005],
            rgba=[0., 1., 0., 0.3],
            visual_only=True,
            static=True)
    ]

    return obj_list


def get_obj_xy_list():
    mid_pos = 0.5
    offset = 0.075
    first_level_y = -0.1
    level_distance = 0.18
    return [
        [mid_pos, first_level_y],
        [mid_pos - offset, first_level_y + level_distance],
        [mid_pos + offset, first_level_y + level_distance],
        [mid_pos - 2 * offset, first_level_y + 2 * level_distance],
        [mid_pos, first_level_y + 2 * level_distance],
        [mid_pos + 2 * offset, first_level_y + 2 * level_distance],
    ]
