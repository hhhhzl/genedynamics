"""
Image preprocessing for MBD3D / 3DGS data adapters.

Dual resolution: infer (128x128) and eval (512x512).
RGBA compositing and normalization.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

try:
    from scipy.ndimage import zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def resize_image(
    img: np.ndarray,
    height: int,
    width: int,
    order: int = 1,
) -> np.ndarray:
    """
    Resize image to (height, width).

    Uses scipy.ndimage.zoom when available, else cv2/pil fallback.
    """
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    if h == height and w == width:
        return img

    if SCIPY_AVAILABLE:
        zy = float(height) / float(h)
        zx = float(width) / float(w)
        if img.ndim == 3:
            return zoom(img, (zy, zx, 1.0), order=order)
        return zoom(img, (zy, zx), order=order)

    try:
        import cv2
        interp = cv2.INTER_LINEAR if order <= 1 else cv2.INTER_CUBIC
        out = cv2.resize(img, (width, height), interpolation=interp)
        return np.asarray(out, dtype=np.float32)
    except ImportError:
        from PIL import Image
        pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        pil_img = pil_img.resize((width, height), Image.BILINEAR)
        return np.asarray(pil_img, dtype=np.float32) / 255.0


def composite_rgba(
    img: np.ndarray,
    background: Literal["white", "black"] = "white",
) -> np.ndarray:
    """
    Composite RGBA to RGB with solid background.

    Args:
        img: (H, W, 3) or (H, W, 4)
        background: "white" or "black"

    Returns:
        (H, W, 3) RGB
    """
    if img.ndim != 3:
        return img[..., :3] if img.ndim == 3 else img

    rgb = img[..., :3]
    if img.shape[-1] < 4:
        return rgb

    alpha = img[..., 3:4]
    if background == "black":
        bg = np.zeros_like(rgb)
    else:
        bg = np.ones_like(rgb)
    return rgb * alpha + bg * (1.0 - alpha)


def normalize_to_01(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] if max > 1."""
    img = np.asarray(img, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def prepare_for_inference(
    img: np.ndarray,
    infer_height: int,
    infer_width: int,
    composite_background: Literal["white", "black"] = "white",
) -> np.ndarray:
    """
    Prepare image for inference: normalize, composite, resize to infer resolution.
    """
    img = normalize_to_01(img)
    img = composite_rgba(img, composite_background)
    img = resize_image(img, infer_height, infer_width)
    return img.astype(np.float32)


def prepare_for_eval(
    img: np.ndarray,
    eval_height: int,
    eval_width: int,
    composite_background: Literal["white", "black"] = "white",
) -> np.ndarray:
    """
    Prepare image for evaluation: normalize, composite, resize to eval resolution.
    """
    img = normalize_to_01(img)
    img = composite_rgba(img, composite_background)
    img = resize_image(img, eval_height, eval_width)
    return img.astype(np.float32)
