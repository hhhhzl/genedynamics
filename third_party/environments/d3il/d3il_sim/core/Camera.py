from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

from environments.d3il.d3il_sim.core.sim_object.sim_object import IntelligentSimObject


class Camera(IntelligentSimObject, ABC):
    """
    Abstract Camera Base Class.
    The Camera looks along its -Z axis
    +X goes to image right, +Y image up.
    """

    def __init__(
        self,
        name: str,
        width: int = 1000,
        height: int = 1000,
        init_pos=None,
        init_quat=None,
        near: float = 0.01,
        far: float = 10,
        fovy: int = 45,
        *args,
        **kwargs
    ):
        """Create a Simulation Camera

        Args:
            name (str): camera name
            width (int): camera image width. Defaults to 1000.
            height (int): camera image height. Defaults to 1000.
            init_pos (vec3, optional): XYZ Position at which the camera spawns. Defaults to None.
            init_quat (vec4, optional): WXYZ Orientation at which the camera spawns. Defaults to None.
            near (float, optional): near focal plane. Defaults to 0.01.
            far (float, optional): far focal plane. Defaults to 10.
            fovy (int, optional): fovy. Defaults to 45. fovx is then calculated using width, height and fovy
        """

        if init_pos is None:
            init_pos = [0, 0, 0]
        if init_quat is None:
            init_quat = [0, 1, 0, 0]

        super(Camera, self).__init__(name, init_pos, init_quat)
        self.width = width
        self.height = height

        self.near = near
        self.far = far
        self.fovy = fovy
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

    def set_cam_params(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        fovy: Optional[int] = None,
    ):
        """Modify Camera parameters
        Default behavior is to keep the existing value if none is passed

        Args:
            width (int): camera image width.
            height (int): camera image height.
            near (float, optional): near focal plane.
            far (float, optional): far focal plane.
            fovy (int, optional): fovy. fovx is then calculated using width, height and fovy
        """

        self.width = width or self.width
        self.height = height or self.height

        self.near = near or self.near
        self.far = far or self.far
        self.fovy = fovy or self.fovy
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

    def get_segmentation(
        self, width: int = None, height: int = None, depth: bool = True
    ) -> np.ndarray:
        """returns a 2D image with object segmentation mask.
        Instead of color (RGB) information, the pixel values are integers corresponding to the depicted object.
        Optionally a depth channel can be also returned


        Args:
            width (int, optional): width of the image. If left unspecified, the camera image width is taken. Defaults to None.
            height (int, optional): width of the image. If left unspecified, the camera image height is taken. Defaults to None.
            depth (bool, optional): if true, return an additional channel with depth information. Defaults to True.

        Returns:
            np.ndarray: a 2D image with segmentation masks and optionally a depth channel
        """

        return self._get_img_data(
            width=width, height=height, depth=depth, segmentation=True
        )

    def get_image(
        self,
        width: int = None,
        height: int = None,
        depth: bool = True,
        denormalize_depth: bool = True,
    ) -> np.ndarray:
        """take an RGB image with this camera

         Args:
            width (int, optional): width of the image. If left unspecified, the camera image width is taken. Defaults to None.
            height (int, optional): width of the image. If left unspecified, the camera image height is taken. Defaults to None.
            depth (bool, optional): if true, return an additional channel with depth information. Defaults to True.

        Returns:
            np.ndarray: a 2D RGB image and optionally a depth channel
        """
        return self._get_img_data(width, height, depth, denormalize_depth, False)

    def calc_point_cloud(
        self, width: int = None, height: int = None, denormalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """calculates a point cloud from this camera's viewpoint

        Args:
            width (int, optional): horizontal resolution. If left unspecified, the camera image width is used. Defaults to None.
            height (int, optional): vertical resolution. If left unspecified, the camera image height is used. Defaults to None.
            denormalize (int, optional): Is the depth channel normalized to [0,1] and must be denormalized? Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: XYZ Coordinates, RGB Colors
        """
        rgb_img, depth_img = self.get_image(
            width, height, denormalize_depth=denormalize
        )

        return self.calc_point_cloud_from_images(rgb_img=rgb_img, depth_img=depth_img)

    def calc_point_cloud_from_images(
        self, rgb_img: np.ndarray, depth_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """calculates a point cloud from camera images
           Also removes invalid points (useful for segmenting depth images for segmented pointclouds)

        Args:
            rgb_img (int, optional): RGB Image as int numpy array with shape (width, height, 3)  (3 for  (r, g, b))
            depth_img (int, optional): Depth Image as float numpy array with shape (width, height, 1) each depth value in meters.

        Returns:
            Tuple[np.ndarray, np.ndarray]: XYZ Coordinates, RGB Colors
        """
        true_width = rgb_img.shape[1]
        true_height = rgb_img.shape[0]
        if self.height == true_height and self.width == true_width:
            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy
        else:
            fx = (true_width / 2) / (np.tan(self.fovx * np.pi / 180 / 2))
            fy = (true_height / 2) / (np.tan(self.fovy * np.pi / 180 / 2))
            cx = true_width / 2
            cy = true_height / 2

        z = depth_img
        u = np.arange(true_width) - cx
        v = np.arange(true_height) - cy

        x = (z * u) / fx
        y = (z.T * v).T / fy

        points = np.stack((x, y, z), axis=-1).reshape((true_width * true_height, 3))
        colors = rgb_img.reshape((true_width * true_height, 3)) / 255.0

        valid_points = ~np.isnan(points).any(axis=1)
        points = points[valid_points]
        colors = colors[valid_points]

        return points, colors

    def denormalize_depth(self, depth_img: np.ndarray) -> np.ndarray:
        """transforms an image from normalized (0-1) depth to actual depth using camera far and near planes

        Args:
            depth_img (np.ndarray): a normalized depth image

        Returns:
            np.ndarray: the denormalized / actual depth image
        """
        # Äquivalent function, but shorter and more often used
        z = self.near / (1 - depth_img * (1 - self.near / self.far))
        return z

    def apply_noise(self, depth_img: np.ndarray) -> np.ndarray:
        """simulates sensor noise for the depth image

        Args:
            depth_img (np.ndarray): a depth image

        Returns:
            np.ndarray: the depth image with added noise
        """
        z = self.denormalize_depth(depth_img)

        # Gaussian noise (scale with power of 2 of the real depth + offset)
        depth_img = depth_img + (
            0.0001 * np.power(z - 0.5, 2) + 0.0004
        ) * np.random.rand(self.height, self.width)

        # Quantization (scale with the inverse depth)
        depth_img = ((40000 * depth_img).astype(int) / 40000.0).astype(np.float32)

        # From normalized to actual depth, with noise
        z = (
            2
            * self.far
            * self.near
            / (self.far + self.near - (self.far - self.near) * (2 * depth_img - 1))
        )

        # Final smoothing with a bilateral filter
        # args: kernel size, sigma in "color" space, sigma in coordinate space
        z = cv2.bilateralFilter(z, 5, 0.1, 5)

        return z

    def get_poi(self) -> list:
        """returns the camera's pois for querying in the scene.
        overwrites IntelligentSimObject.get_poi()
        Returns:
            list: list of pois
        """
        return [self.name]

    @abstractmethod
    def _get_img_data(
        self,
        width: int = None,
        height: int = None,
        depth: bool = True,
        denormalize_depth: bool = True,
        segmentation: bool = False,
    ) -> np.ndarray:
        """private abstract method for taking the raw image data.
        This abstract method gets exposed by the various public methods of the Camera Base Class
        and must to be implemented by the concrete child for each simulator.

        Args:
            width (int, optional): width of the image. If left unspecified, the camera image width is taken. Defaults to None.
            height (int, optional): width of the image. If left unspecified, the camera image height is taken. Defaults to None.
            depth (bool, optional): if true, return an additional channel with depth information. Defaults to True.
            segmentation (bool, optional): if true, take segmentation mask instead of RGB colors. Defaults to False.

        Returns:
            np.ndarray: raw image data
        """
        pass

    @abstractmethod
    def get_cart_pos_quat(self) -> Tuple[np.ndarray, np.ndarray]:
        """abstract method for reading the cameras cartesion position and quaternion.
        Useful for calculating 3D coordinates. This abstract method must be implemented by the concrete child for each simulator.

        Returns:
            Tuple[np.ndarray, np.ndarray]: XYZ Position, WXYZ Quaternion
        """
        pass

    @property
    def fov(self):
        """deprecated not precisely defined property, still used by some simulations, mostly reflects the fovy"""
        return self.fovy
