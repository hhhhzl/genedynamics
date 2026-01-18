import pybullet as p

from environments.d3il.d3il_sim.utils.geometric_transformation import wxyz_to_xyzw


def pb_load(prim_obj, *args, **kwargs) -> int:
    sim = args[0]

    pb_kwargs = {}

    if prim_obj.type == prim_obj.Shape.BOX:
        pb_kwargs["shapeType"] = p.GEOM_BOX
        pb_kwargs["halfExtents"] = prim_obj.size
    elif prim_obj.type == prim_obj.Shape.SPHERE:
        pb_kwargs["shapeType"] = p.GEOM_SPHERE
        pb_kwargs["radius"] = prim_obj.size[0]
    elif prim_obj.type == prim_obj.Shape.CYLINDER:
        pb_kwargs["shapeType"] = p.GEOM_CYLINDER
        pb_kwargs["radius"] = prim_obj.size[0]
        pb_kwargs["height"] = prim_obj.size[1]

    coll_id = -1  # Disables Collision
    if not prim_obj.visual_only:
        coll_id = p.createCollisionShape(**pb_kwargs, physicsClientId=sim)
    vis_id = p.createVisualShape(
        **pb_kwargs, rgbaColor=prim_obj.rgba, physicsClientId=sim
    )
    if prim_obj.mass:
        obj_id = p.createMultiBody(
            baseMass=prim_obj.mass,
            baseCollisionShapeIndex=coll_id,
            baseVisualShapeIndex=vis_id,
            basePosition=prim_obj.init_pos,
            baseOrientation=wxyz_to_xyzw(prim_obj.init_quat),
            physicsClientId=sim,
        )
    else:
        obj_id = p.createMultiBody(
            baseCollisionShapeIndex=coll_id,
            baseVisualShapeIndex=vis_id,
            basePosition=prim_obj.init_pos,
            baseOrientation=wxyz_to_xyzw(prim_obj.init_quat),
            physicsClientId=sim,
        )

    return obj_id
