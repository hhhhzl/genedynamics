import os
import xml.etree.ElementTree as Et
from typing import Tuple

from environments.d3il.d3il_sim.sims.mujoco import MujocoLoadable as mj_load
from environments.d3il.d3il_sim.utils import sim_path


class MujocoObject(mj_load.MujocoLoadable):
    def __init__(
        self, object_name, pos, quat, root=sim_path.D3IL_DIR, obj_path=None
    ):

        if pos is None:
            pos = [0, 0, 0]
        else:
            assert len(pos) == 3, "Error, parameter pos has to be three dimensional."

        if quat is None:
            quat = [0, 0, 0, 0]
        else:
            assert len(quat) == 4, "Error, parameter quat has to be three dimensional."

        if obj_path is None:
            obj_path = os.path.join(
                root, "./simulation/envs/mujoco/assets/" + object_name + ".xml"
            )
        self.obj_path = obj_path
        self.pos = pos
        self.quat = quat
        self.name = object_name

    def to_mj_xml(self, scene_dir: str) -> Tuple[Et.Element, bool]:
        obj = Et.parse(self.obj_path)
        worldbody = obj.find("worldbody")
        body = worldbody.find("body")

        # cast types to string for xml parsing
        obj_pos_str = " ".join(map(str, self.pos))
        obj_quat_str = " ".join(map(str, self.quat))

        body.set("pos", obj_pos_str)
        body.set("quat", obj_quat_str)

        obj.write(self.obj_path)

        include = Et.Element("include")
        include.set("file", os.path.relpath(self.obj_path, scene_dir))
        return include, True


class MujocoWorkspace(mj_load.MujocoXmlLoadable):
    def __init__(self, size: str):
        if size not in ["small", "medium", "large"]:
            raise ValueError(
                "Error, please choose a size between <small>, <medium> or <large> ."
            )
        self.size = size

    @property
    def xml_file_path(self):
        return sim_path.d3il_path(
            "./envs/mujoco/assets/workspace/{}_workspace.xml".format(self.size)
        )


class MujocoSurrounding(mj_load.MujocoXmlLoadable):
    def __init__(self, xml_path):
        self.xml_path = xml_path

    @property
    def xml_file_path(self):
        return self.xml_path
