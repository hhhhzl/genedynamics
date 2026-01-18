import os
import xml.etree.ElementTree as Et
from typing import Tuple

import numpy as np
import yaml

import environments.d3il.d3il_sim.sims.mj_beta.MjLoadable as mj_load
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import IncludeType
from environments.d3il.d3il_sim.utils import sim_path


class MujocoObject(mj_load.MjXmlLoadable):
    def __init__(self, object_name, pos, quat, root=sim_path.D3IL_DIR):
        if pos is None:
            pos = [0, 0, 0]
        else:
            assert len(pos) == 3, "Error, parameter pos has to be three dimensional."

        if quat is None:
            quat = [0, 0, 0, 0]
        else:
            assert len(quat) == 4, "Error, parameter quat has to be four dimensional."

        self.obj_dir_path = "./models/mj/common-objects/" + object_name
        self.obj_file_name = object_name + ".xml"
        self.root = root
        self.pos = pos
        self.quat = quat
        self.name = object_name
        super().__init__(os.path.join(root, self.obj_dir_path, self.obj_file_name))

    def mj_load(self) -> Tuple[Et.Element, list, IncludeType]:
        include_et, xml, is_include = super(MujocoObject, self).mj_load()
        xml_file = xml[0]  # in this case we only expect one file
        obj = Et.ElementTree(Et.fromstring(xml_file))
        worldbody = obj.find("worldbody")
        body = worldbody.find("body")

        # cast types to string for xml parsing
        obj_pos_str = " ".join(map(str, self.pos))
        obj_quat_str = " ".join(map(str, self.quat))

        body.set("pos", obj_pos_str)
        body.set("quat", obj_quat_str)

        return include_et, Et.tostring(obj), IncludeType.FILE_INCLUDE

    def get_poi(self) -> list:
        """

        Returns:
            a list of points of interest for the scene to query
        """
        return [self.name]


class CompositeMujocoObject(mj_load.MjLoadable, mj_load.MjFreezable):
    def __init__(self, object_folder, object_name, pos, quat):
        if pos is None:
            pos = [0, 0, 0]
        else:
            assert len(pos) == 3, "Error, parameter pos has to be three dimensional."

        if quat is None:
            quat = [0, 0, 0, 0]
        else:
            assert len(quat) == 4, "Error, parameter quat has to be four dimensional."

        self.object_folder = object_folder
        self.pos = pos
        self.quat = quat
        self.name = object_name
        self.file_name = f"{self.name}.xml"

    def mj_load(self) -> Tuple[Et.Element, list, bool]:
        et_include = Et.Element("include")
        et_include.set("file", self.file_name)

        xml_string = self.generate_xml()
        return (
            et_include,
            {os.path.join(self.object_folder, self.file_name): xml_string},
            IncludeType.FILE_INCLUDE,
        )

    def generate_xml(self) -> str:
        info_file = os.path.join(self.object_folder, "info.yml")
        assert os.path.isfile(
            info_file
        ), f"The file {info_file} was not found. Did you specify the path to the object folder correctly?"

        with open(info_file, "r") as f:
            info_dict = yaml.safe_load(f)

        original_file = info_dict["original_file"]
        submesh_files = info_dict["submesh_files"]
        submesh_props = info_dict["submesh_props"]
        weight = info_dict["weight"]
        material_map = info_dict["material_map"]

        root = Et.Element("mujoco", attrib={"model": self.name})

        # Assets and Worldbody
        assets = Et.SubElement(root, "asset")
        worldbody = Et.SubElement(root, "worldbody")
        body_attributes = {
            "name": f"{self.name}",
            "pos": " ".join(map(str, self.pos)),
            "quat": " ".join(map(str, self.quat)),
        }
        body = Et.SubElement(worldbody, "body", attrib=body_attributes)

        ## Texture and Material
        texture_attributes = {
            "type": "2d",
            "name": f"{self.name}_tex",
            "file": os.path.join(self.object_folder, material_map),
        }
        texture = Et.SubElement(assets, "texture", attrib=texture_attributes)

        material_attributes = {
            "name": f"{self.name}_mat",
            "texture": texture_attributes["name"],
            "specular": "0.5",
            "shininess": "0.5",
        }
        material = Et.SubElement(assets, "material", attrib=material_attributes)

        # Meshes
        orig_mesh_attributes = {
            "name": f"{self.name}_orig",
            "file": os.path.join(self.object_folder, original_file),
        }
        orig_mesh = Et.SubElement(assets, "mesh", attrib=orig_mesh_attributes)

        orig_geom_attributes = {
            "material": material_attributes["name"],
            "mesh": orig_mesh_attributes["name"],
            "group": "2",
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
        }
        orig_geom = Et.SubElement(body, "geom", attrib=orig_geom_attributes)

        for i, (submesh_file, submesh_prop) in enumerate(
            zip(submesh_files, submesh_props)
        ):
            collision_mesh_attributes = {
                "name": f"{self.name}_coll_{i}",
                "file": os.path.join(self.object_folder, submesh_file),
            }
            collision_mesh = Et.SubElement(
                assets, "mesh", attrib=collision_mesh_attributes
            )
            collision_geom_attributes = {
                "mesh": collision_mesh_attributes["name"],
                "mass": str(weight * submesh_prop),
                "group": "3",
                "type": "mesh",
                "conaffinity": "1",
                "contype": "1",
                "condim": "4",
                "friction": "0.95 0.3 0.1",
                "rgba": "1 1 1 1",
                "solimp": "0.998 0.998 0.001",
                "solref": "0.001 1",
            }
            collision_geom = Et.SubElement(
                body, "geom", attrib=collision_geom_attributes
            )

        joint_attributes = {
            "damping": "0.0001",
            "name": f"{self.name}:joint",
            "type": "free",
        }
        joint = Et.SubElement(body, "joint", attrib=joint_attributes)

        return Et.tostring(root)

    def get_poi(self) -> list:
        """

        Returns:
            a list of points of interest for the scene to query
        """
        return [self.name]

    def freeze(self, model, data):
        pos: np.ndarray = data.jnt(self.name + ":joint").qpos
        self.pos = pos[0:3].tolist()
        self.quat = pos[3:].tolist()

    def unfreeze(self, data, model):
        pass


class YCBMujocoObject(CompositeMujocoObject):
    def __init__(self, ycb_base_folder, object_id, object_name, pos, quat):
        self.ycb_base_folder = ycb_base_folder
        self.ycb_object_folder = os.path.join(ycb_base_folder, object_id)

        super().__init__(self.ycb_object_folder, object_name, pos, quat)


class CustomMujocoObject(mj_load.MjXmlLoadable):
    def __init__(
        self, object_name, object_dir_path, pos, quat, root=sim_path.D3IL_DIR
    ):
        if pos is None:
            pos = [0, 0, 0]
        else:
            assert len(pos) == 3, "Error, parameter pos has to be three dimensional."

        if quat is None:
            quat = [0, 0, 0, 0]
        else:
            assert len(quat) == 4, "Error, parameter quat has to be four dimensional."

        self.obj_dir_path = object_dir_path
        self.obj_file_name = object_name + ".xml"
        self.root = root
        self.pos = pos
        self.quat = quat
        self.name = object_name

        super().__init__(os.path.join(root, self.obj_dir_path, self.obj_file_name))

    def mj_load(self) -> Tuple[Et.Element, list, bool]:
        include_et, xml, is_include = super(CustomMujocoObject, self).mj_load()
        xml_file = list(xml.values())[0]  # in this case we only expect one file
        obj = Et.ElementTree(Et.fromstring(xml_file))
        worldbody = obj.find("worldbody")
        body = worldbody.find("body")

        # cast types to string for xml parsing
        obj_pos_str = " ".join(map(str, self.pos))
        obj_quat_str = " ".join(map(str, self.quat))

        body.set("pos", obj_pos_str)
        body.set("quat", obj_quat_str)

        return (
            include_et,
            {
                os.path.join(self.loadable_dir, self.file_name): Et.tostring(
                    obj.getroot()
                )
            },
            is_include,
        )

    def get_poi(self) -> list:
        """

        Returns:
            a list of points of interest for the scene to query
        """
        return [self.name]


class MjWorkspace(mj_load.MjXmlLoadable):
    def __init__(self, size: str, root=sim_path.D3IL_DIR):
        if size not in ["small", "medium", "large"]:
            raise ValueError(
                "Error, please choose a size between <small>, <medium> or <large> ."
            )
        self.size = size
        self.root = root
        super().__init__(
            os.path.join(root, "./models/mj/workspace/{}".format(self.size), "ws.xml")
        )


class MjSurrounding(mj_load.MjXmlLoadable):
    def __init__(self, surrounding_name, root=None):
        self.surrounding_name = surrounding_name
        if root is None:
            root = os.path.join(sim_path.D3IL_DIR, "./models/mj/surroundings")
        self.root = root

        super().__init__(os.path.join(root, "{}.xml".format(self.surrounding_name)))
