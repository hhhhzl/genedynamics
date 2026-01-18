"""
The real robots in our lab have a small offset of their TCP compared to the Simulation models,
due to the 3D printed plate for inhand cameras.
"""
import numpy as np

from environments.d3il.d3il_sim.core import RobotBase
from environments.d3il.d3il_sim.utils import geometric_transformation as geom

# TCP is shifted by 5mm in global Z axis
OFFSET_POS = np.array([0.0, 0.0, 0.005])
OFFSET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def corrected_pos_quat(
    robot: RobotBase = None, offset_pos=OFFSET_POS, offset_quat=OFFSET_QUAT
):
    tcp_pos = robot.current_c_pos
    tcp_quat = robot.current_c_quat

    true_tcp = tcp_pos + geom.quat_rot_vec(tcp_quat, offset_pos)
    return true_tcp, geom.quat_mul(tcp_quat, offset_quat)


def adjusted_target_pos(target_pos, target_quat, offset_pos=OFFSET_POS):
    rotated_offset = geom.quat_rot_vec(target_quat, offset_pos)
    true_target_pos = target_pos - rotated_offset
    return true_target_pos
