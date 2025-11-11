"""
RGBA Mask Generation Service
Creates RGBA overlays for facial regions based on segmentation masks.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def resize_mask_to_image(mask: np.ndarray, image_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize mask to match the spatial dimensions of the image.

    Args:
        mask: Mask array (H, W)
        image_shape: Shape of the target image (H, W, C)

    Returns:
        Resized mask with values in {0, 1}
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    target_h, target_w = image_shape[:2]
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return (mask > 0.5).astype(np.uint8)

    resized = cv2.resize(
        mask.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0.5).astype(np.uint8)


def compute_average_color(
    image_rgb: np.ndarray,
    region_mask: np.ndarray,
) -> Optional[List[int]]:
    """
    Compute the average RGB color of the pixels inside the region mask.

    Args:
        image_rgb: H x W x 3 uint8 image array (RGB order)
        region_mask: H x W mask with values 0 or 1 (float/bool/int)

    Returns:
        [R, G, B] list or None if the mask is empty.
    """
    if image_rgb.size == 0 or region_mask.size == 0:
        return None

    mask_bool = region_mask.astype(bool)
    if not np.any(mask_bool):
        return None

    region_pixels = image_rgb[mask_bool]
    if region_pixels.size == 0:
        return None

    avg = region_pixels.mean(axis=0).astype(np.uint8)
    return avg.tolist()


def create_rgba_overlay(
    image_rgb: np.ndarray,
    region_mask: np.ndarray,
    fill_with_original: bool = True,
    fallback_color: Optional[Iterable[int]] = None,
) -> Image.Image:
    """
    Create an RGBA overlay image for a segmented region.

    Args:
        image_rgb: H x W x 3 uint8 image array (RGB)
        region_mask: H x W mask with values 0 or 1 (float/bool/int)
        fill_with_original: When True, RGB channels use the original pixels inside the mask.
                            Otherwise, a single fallback color is used.
        fallback_color: 3-element iterable in RGB order. Used when fill_with_original=False
                        or when the region mask is empty.

    Returns:
        PIL.Image in RGBA mode.
    """
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8 array")

    if region_mask.ndim != 2:
        raise ValueError("region_mask must be 2D")

    h, w = region_mask.shape
    if image_rgb.shape[0] != h or image_rgb.shape[1] != w:
        image_aligned = cv2.resize(
            image_rgb.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        image_aligned = image_rgb
    mask_bool = region_mask.astype(bool)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if np.any(mask_bool):
        if fill_with_original:
            rgba[..., :3][mask_bool] = image_aligned[mask_bool]
        else:
            color = np.array(
                fallback_color if fallback_color is not None else [0, 0, 0],
                dtype=np.uint8,
            )
            rgba[..., :3][mask_bool] = color

        rgba[..., 3][mask_bool] = 255
    else:
        # No region pixels; if fallback_color is provided we fill uniformly with alpha 0
        if fallback_color is not None:
            rgba[..., :3] = np.array(fallback_color, dtype=np.uint8)

    return Image.fromarray(rgba, mode="RGBA")


def generate_region_overlays(
    image_rgb: np.ndarray,
    region_masks: Dict[str, np.ndarray],
    fill_with_original: bool = True,
    fallback_colors: Optional[Dict[str, Iterable[int]]] = None,
) -> Dict[str, Image.Image]:
    """
    Generate RGBA overlays for all provided regions.

    Args:
        image_rgb: H x W x 3 uint8 image array
        region_masks: Mapping of region name -> mask array (H x W)
        fill_with_original: Whether to use original pixels inside the mask
        fallback_colors: Optional mapping region -> RGB iterable when fill_with_original=False

    Returns:
        Mapping of region name -> PIL.Image (RGBA)
    """
    overlays: Dict[str, Image.Image] = {}
    fallback_colors = fallback_colors or {}

    for region_name, mask in region_masks.items():
        overlays[region_name] = create_rgba_overlay(
            image_rgb=image_rgb,
            region_mask=mask,
            fill_with_original=fill_with_original,
            fallback_color=fallback_colors.get(region_name),
        )

    return overlays

