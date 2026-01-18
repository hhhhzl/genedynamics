from typing import Tuple
from xml.etree import ElementTree as Et

import numpy as np
from mujoco import MjData, MjModel, mj_name2id, mjtObj

from environments.d3il.d3il_sim.core.Camera import Camera
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import IncludeType
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_render_singleton import render
from environments.d3il.d3il_sim.sims.mj_beta.MjLoadable import MjLoadable
from environments.d3il.d3il_sim.utils.geometric_transformation import (
    mat2posQuat,
    posRotMat2TFMat,
    quat2mat,
)


class MjCamera(Camera, MjLoadable):
    def __init__(
        self,
        name: str,
        width: int = 1000,
        height: int = 1000,
        init_pos=None,
        init_quat=None,
        near=0.01,
        far=10,
        *args,
        **kwargs,
    ):
        super(MjCamera, self).__init__(
            name, width, height, init_pos, init_quat, near=near, far=far
        )
        self.viewer = None
        self.data = None
        self.model = None

    def register_sim(self, sim: Tuple[MjModel, MjData], sim_name: str):
        self.model = sim[0]
        self.data = sim[1]
        super().register_sim(sim, sim_name)

        # Gathering the intrinsics used by Mujoco for the cameras
        extent = self.model.stat.extent
        self.near = self.model.vis.map.znear * extent
        self.far = self.model.vis.map.zfar * extent

        cam_id = mj_name2id(m=self.model, name=self.name, type=mjtObj.mjOBJ_CAMERA)
        self.fovy = self.model.cam_fovy[cam_id]
        self.fovx = (
            2
            * np.arctan(
                self.width
                * 0.5
                / (self.height * 0.5 / np.tan(self.fovy * np.pi / 360 / 2))
            )
            / np.pi
            * 360
        )

        self.fx = (self.width / 2) / (np.tan(self.fovx * np.pi / 180 / 2))
        self.fy = (self.height / 2) / (np.tan(self.fovy * np.pi / 180 / 2))
        self.cx = self.width / 2
        self.cy = self.height / 2

        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
        )

        return

    def _get_img_data(
        self,
        width: int = None,
        height: int = None,
        depth: bool = True,
        denormalize_depth: bool = True,
        segmentation: bool = False,
    ):
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        rendered_images = render(
            cam_name=self.name,
            width=width,
            height=height,
            depth=depth,
            segmentation=segmentation,
            model=self.model,
            data=self.data,
        )

        if depth:
            color_img, depth_img = rendered_images
            color_img = self.vertical_flip(color_img)
            depth_img = self.vertical_flip(depth_img)
            if denormalize_depth:
                depth_img = self.denormalize_depth(depth_img)
            return color_img, depth_img
        elif segmentation:
            seg_img = rendered_images
            seg_img = self.vertical_flip(seg_img)
            return seg_img
        else:
            color_img = rendered_images
            color_img = self.vertical_flip(color_img)
            return color_img

        # return mj_render.render(self.sim, self.name, width, height, depth, segmentation)

    def get_cart_pos_quat(self):
        cam_id = mj_name2id(m=self.model, name=self.name, type=mjtObj.mjOBJ_CAMERA)

        cam_pos = self.data.cam_xpos[cam_id]
        cam_mat = self.data.cam_xmat[cam_id]
        c2b_r = cam_mat.reshape((3, 3))
        # In MuJoCo, we assume that a camera is specified in XML as a body
        #    with pose p, and that that body has a camera sub-element
        #    with pos and euler 0.
        #    Therefore, camera frame with body euler 0 must be rotated about
        #    x-axis by 180 degrees to align it with the world frame.
        b2w_r = quat2mat([0, 1, 0, 0])
        c2w_r = np.matmul(c2b_r, b2w_r)
        c2w = posRotMat2TFMat(cam_pos, c2w_r)
        pos, quat = mat2posQuat(c2w)

        return pos, quat

    def mj_load(self):
        # cast types to string for xml parsing
        pos_str = " ".join(map(str, self.init_pos))
        quat_str = " ".join(map(str, self.init_quat))

        # Create new object for xml tree
        object_body = Et.Element("body")
        object_body.set("pos", pos_str)
        object_body.set("quat", quat_str)
        object_body.set("name", self.name)

        cam = Et.SubElement(object_body, "camera")
        cam.set("name", self.name)
        cam.set("fovy", str(self.fovy))

        return object_body, {}, IncludeType.WORLD_BODY

    def get_segmentation(self, width=None, height=None, depth=True):
        data = super(MjCamera, self).get_segmentation(width, height, depth)

        if depth:
            data, d = data

        obj_types = data[..., 0]  # Might be useful in the future
        obj_ids = self.model.geom_bodyid[data[..., 1]]

        if depth:
            return obj_ids, d
        return obj_ids

    def vertical_flip(self, img):
        return np.flipud(img)


class MjInhandCamera(MjCamera):
    """
    In hand camera of the robot. Extends camera base class.
    """

    def __init__(
        self,
        robot_cam_id: str,
        width: int = 96,
        height: int = 96,
        *args,
        **kwargs,
    ):
        super().__init__(
            robot_cam_id,
            width,
            height,
            *args,
            **kwargs,
        )

    def mj_load(self):
        """
        We assume this kind of camera is included in another XML file.
        Hence, we do not need to create a new object for the XML tree.
        """
        return None, None, IncludeType.VIRTUAL_INCLUDE


class MjCageCam(MjCamera):
    """
    Cage camera. Extends the camera base class.
    """

    def __init__(self, width: int = 1000, height: int = 1000, *args, **kwargs):
        super().__init__(
            "rgbd_cage",
            width,
            height,
            init_pos=[0.7, 0.1, 0.9],
            init_quat=[
                0.6830127,
                0.1830127,
                0.1830127,
                0.683012,
            ],  # Looking with 30 deg to the robot
            *args,
            **kwargs,
        )
