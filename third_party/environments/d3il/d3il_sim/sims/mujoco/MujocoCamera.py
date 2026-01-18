from typing import Tuple
from xml.etree import ElementTree as Et

import numpy as np

from environments.d3il.d3il_sim.core.Camera import Camera
from environments.d3il.d3il_sim.sims.mujoco.mj_utils import mujoco_render_singleton as mj_render
from environments.d3il.d3il_sim.sims.mujoco.MujocoLoadable import MujocoLoadable
from environments.d3il.d3il_sim.utils.geometric_transformation import (
    mat2posQuat,
    mat2quat,
    posRotMat2TFMat,
    quat2mat,
)


class MujocoCamera(Camera, MujocoLoadable):
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
        super(MujocoCamera, self).__init__(
            name, width, height, init_pos, init_quat, near=near, far=far
        )
        self.viewer = None

    def register_sim(self, sim, sim_name: str):
        super().register_sim(sim, sim_name)

        # Gathering the intrinsics used by Mujoco for the cameras
        extent = self.sim.model.stat.extent
        self.near = self.sim.model.vis.map.znear * extent
        self.far = self.sim.model.vis.map.zfar * extent

        cam_id = sim.model._camera_name2id[self.name]
        fovy = sim.model.cam_fovy[cam_id]
        self.set_cam_params(fovy=fovy)

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

        rendered_images = mj_render.render(
            self.sim, self.name, width, height, depth, segmentation
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
        cam_pos = self.sim.data.get_camera_xpos(self.name)
        cam_mat = self.sim.data.get_camera_xmat(self.name)
        c2b_r = cam_mat
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

    def to_mj_xml(self, scene_dir: str) -> Tuple[Et.Element, bool]:
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

        return object_body, False

    def get_segmentation(self, width=None, height=None, depth=True):
        data = super(MujocoCamera, self).get_segmentation(width, height, depth)
        return data[:, :, 1]

    def vertical_flip(self, img):
        return np.flip(img, axis=0)


class MjInhandCamera(MujocoCamera):
    """
    In hand camera of the robot. Extends camera base class.
    """

    def __init__(
        self,
        robot_cam_id: str,
        width: int = 1000,
        height: int = 1000,
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


class MjCageCam(MujocoCamera):
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
