import xml.etree.ElementTree as Et

from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import IncludeType


def mj_load(prim_obj, *args, **kwargs):
    # cast types to string for xml parsing
    pos_str = " ".join(map(str, prim_obj.init_pos))
    orientation_str = " ".join(map(str, prim_obj.init_quat))
    mass_str = str(prim_obj.mass)
    size_str = " ".join(map(str, prim_obj.size))
    rgba_str = " ".join(map(str, prim_obj.rgba))

    # Create new object for xml tree
    object_body = Et.Element("body")
    object_body.set("name", prim_obj.name)
    object_body.set("pos", pos_str)
    object_body.set("quat", orientation_str)

    geom = Et.SubElement(object_body, "geom")
    geom.set("type", prim_obj.type.value)
    geom.set("name", "{}:geom".format(prim_obj.name))
    if prim_obj.mass:
        geom.set("mass", mass_str)
    geom.set("size", size_str)
    geom.set("rgba", rgba_str)

    if prim_obj.visual_only:
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    # Set object parameters
    if not prim_obj.static:
        Et.SubElement(object_body, "freejoint")

    if prim_obj.solimp is not None:
        geom.set("solimp", " ".join(prim_obj.solimp))
    if prim_obj.solref is not None:
        geom.set("solref", " ".join(prim_obj.solref))

    inc_type = IncludeType.WORLD_BODY
    return object_body, {}, inc_type
