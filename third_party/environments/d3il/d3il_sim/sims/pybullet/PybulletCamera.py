import warnings

import numpy as np
import pybullet as p

from environments.d3il.d3il_sim.core.Camera import Camera
from environments.d3il.d3il_sim.sims.pybullet.PybulletLoadable import PybulletLoadable
from environments.d3il.d3il_sim.utils.geometric_transformation import wxyz_to_xyzw, xyzw_to_wxyz


class PybulletCamera(Camera, PybulletLoadable):
    def __init__(
        self,
        name: str,
        width: int = 1000,
        height: int = 1000,
        init_pos=None,
        init_quat=None,
        cam_link_name="camera_color_optical_frame",
        *args,
        **kwargs,
    ):
        super(PybulletCamera, self).__init__(
            name, width, height, init_pos, init_quat, *args, **kwargs
        )

        self.cam_link_name = cam_link_name

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

        view_matrix = self._get_view_matrix()
        projection_matrix = self._get_projection_matrix(width, height)

        _, _, rgb_img, depth_img, seg_img = p.getCameraImage(
            width,
            height,
            view_matrix,
            projection_matrix,
            physicsClientId=self.sim,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_img = rgb_img[:, :, :3]
        if segmentation:
            rgb_img = seg_img

        if depth:
            if denormalize_depth:
                depth_img = self.denormalize_depth(depth_img)
            return rgb_img, depth_img
        return rgb_img

    def _get_view_matrix(self):
        init_camera_vector = (0, 0, 1)  # z axis
        init_up_vector = (0, -1, 0)

        pos, quat = self.get_cart_pos_quat_internal()
        rot_matrix = p.getMatrixFromQuaternion(quat)
        rot_matrix = np.array(rot_matrix).reshape((3, 3))
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        camera_eye_pos = np.array(pos)
        camera_target_position = camera_eye_pos + 0.2 * camera_vector

        view_matrix = p.computeViewMatrix(
            camera_eye_pos, camera_target_position, up_vector
        )
        return view_matrix

    def _get_projection_matrix(self, width: int, height: int):
        aspect = width / height

        return p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

    def get_cart_pos_quat_internal(self):
        return self.init_pos, self.init_quat

    def get_cart_pos_quat(self):
        pos, quat = self.get_cart_pos_quat_internal()
        return pos, xyzw_to_wxyz(quat)

    def pb_load(self, pb_sim) -> int:
        vis_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, rgbaColor=[0, 0, 0, 0], physicsClientId=pb_sim
        )
        return vis_id


class PbInHandCamera(PybulletCamera):
    """
    In hand camera of the robot. Extends camera base class.
    """

    def __init__(self, width: int = 1000, height: int = 1000, *args, **kwargs):
        super().__init__(
            f"rgbd_rb{self.GLOBAL_NAME_COUNTER}",
            width,
            height,
            *args,
            **kwargs,
        )
        self.robot_id = None

    def set_robot_id(self, rb_id):
        self.robot_id = rb_id

    def get_cart_pos_quat_internal(self):
        if self.robot_id is None:
            warnings.warn("Cannot use InhandCam without a robot.")

        # Seemingly the only way to find the link id of a specifically named link
        cam_link_id = -1
        for link_id in range(p.getNumJoints(self.robot_id)):
            link_name = p.getJointInfo(self.robot_id, link_id)[12].decode("UTF-8")
            if link_name == self.cam_link_name:
                cam_link_id = link_id
                break

        if cam_link_id == -1:
            warnings.warn(f"Cannot find the camera link {self.cam_link_name}!")

        pos, quat, _, _, _, _ = p.getLinkState(
            self.robot_id,
            linkIndex=cam_link_id,
            computeForwardKinematics=True,
            physicsClientId=self.sim,
        )
        return pos, quat


class PbCageCam(PybulletCamera):
    """
    Cage camera. Extends the camera base class.
    """

    def __init__(self, width: int = 1000, height: int = 1000, *args, **kwargs):
        super().__init__(
            "rgbd_cage", width, height, init_pos=[0.7, 0.1, 0.9], *args, **kwargs
        )
