from typing import Tuple

import numpy as np

from environments.d3il.d3il_sim.core.Camera import Camera


class SlCamera(Camera):
    def _get_img_data(
        self,
        width: int = None,
        height: int = None,
        depth: bool = True,
        denormalize_depth: bool = True,
        segmentation: bool = False,
    ) -> np.ndarray:
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        return np.zeros((width, height, 3 + depth))

    def get_cart_pos_quat(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.zeros(4)
