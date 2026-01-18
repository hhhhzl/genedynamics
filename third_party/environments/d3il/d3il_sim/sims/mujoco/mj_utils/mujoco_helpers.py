import logging

import mujoco_py
import numpy as np


def get_mj_geom_id(obj_name, mj_model):
    obj_id = mj_model.geom_name2id(obj_name)

    if obj_id == -1:
        logging.getLogger(__name__).warning(
            "Could not find geom for obj1 with name {}".format(obj_name)
        )

    return obj_id


def has_collision(mj_scene, obj1_name: str, obj2_name: str = None) -> bool:
    """Check if a collision with obj1 occured.
    If obj2_name is unspecified, any collision will count, else only collisions between the two specified objects.

    Args:
        mj_scene (_type_): Mujoco Scene
        obj1_name (str): name of the geom to be checked
        obj2_name (str, optional): name of a second geom. If undefined, all collisions are considered valid. Defaults to None.

    Returns:
        bool: true if a collision occured, false otherwise
    """
    obj1 = get_mj_geom_id(obj1_name, mj_scene.sim.model)
    if obj1 == -1:
        return False

    if obj2_name is not None:
        obj2 = get_mj_geom_id(obj2_name, mj_scene.sim.model)
        if obj2 == -1:
            return False

    for i in range(mj_scene.sim.data.ncon):
        contact = mj_scene.sim.data.contact[i]
        if contact.geom1 == obj1 or contact.geom2 == obj1:
            if obj2_name is None:
                return True
            elif contact.geom1 == obj2 or contact.geom2 == obj2:
                return True
    return False


def change_domain(value, in_low, in_high, out_low, out_high):
    return (out_high - out_low) * ((value - in_low) / (in_high - in_low)) + out_low


def obj_position(sim, obj_id: str):
    return sim.data.get_body_xpos(obj_id)


def obj_distance(sim, obj1: str, obj2: str):
    obj1_pos = obj_position(sim, obj1)
    obj2_pos = obj_position(sim, obj2)
    dist = np.linalg.norm(obj1_pos - obj2_pos)
    rel_dist = (obj1_pos - obj2_pos) / dist + 1e-8
    return dist, rel_dist


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (
        sim.model.eq_type is None
        or sim.model.eq_obj1id is None
        or sim.model.eq_obj2id is None
    ):
        return
    for eq_type, obj1_id, obj2_id in zip(
        sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id
    ):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]
