import numpy as np


def obj_distance(obj1_pos, obj2_pos):
    dist = np.linalg.norm(obj1_pos - obj2_pos)
    rel_dist = (obj1_pos - obj2_pos) / dist + 1e-8
    return dist, rel_dist
