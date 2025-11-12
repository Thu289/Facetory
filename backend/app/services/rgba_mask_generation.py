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


def attenuate_skin_overlay_alpha(
    overlay: Image.Image,
    detail_strength: Optional[List[Tuple[float, float]]] = None,
    highlight_percentile: float = 97.5,
    min_highlight_luminance: float = 0.85,
    highlight_scale: float = 0.35,
    highlight_blur_sigma: float = 2.5,
    diff_gamma: float = 1.35,
    base_alpha_floor: float = 0.35,
    base_alpha_scale: float = 0.65,
    blend_mode: str = "linear",
    softlight_strength: float = 0.4,
) -> Image.Image:
    """Reduce alpha for skin pixels that match the dominant skin tone.

    Args:
        overlay: Skin RGBA overlay image.
        detail_strength: Optional list of (normalized_threshold, alpha_weight) tuples in ascending order.
        highlight_percentile: Percentile (0-100) of luminance used to detect highlights for attenuation.
        min_highlight_luminance: Minimum luminance (0-1 range) to classify a pixel as highlight.
        highlight_scale: Minimum multiplier (0-1) applied to alpha in highlight regions.
        highlight_blur_sigma: Gaussian sigma (in pixels) applied to highlight mask for soft reduction.
        diff_gamma: Gamma exponent (>0) applied to normalized color distance before weighting.
        base_alpha_floor: Minimum fraction of original alpha to preserve across the region (0-1).
        base_alpha_scale: Additional alpha scaling applied after floor (0-1).
        blend_mode: Optional blend mode (`linear`, `softlight`) used when reconstructing detail map.
        softlight_strength: Strength (0-1) used when `blend_mode="softlight"`.

    Returns:
        Processed overlay preserving only detail regions; original overlay if no change.
    """

    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")

    rgba = np.array(overlay, dtype=np.uint8)
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    mask = alpha > 1e-3

    if not np.any(mask):
        return overlay

    rgb = rgba[..., :3].astype(np.float32)
    samples = rgb[mask]

    if samples.size == 0:
        return overlay

    base_color = np.median(samples, axis=0)
    dominant_color = base_color

    try:
        data = samples.astype(np.float32)
        if data.shape[0] >= 3:
            K = 3
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
            ret, labels, centers = cv2.kmeans(data, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            if centers is not None and len(centers) > 0:
                base_lab = cv2.cvtColor(base_color.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2Lab).reshape(3)
                center_lab = cv2.cvtColor(centers.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2Lab).reshape(-1, 3)
                distances = np.linalg.norm(center_lab - base_lab, axis=1)
                dominant_idx = int(np.argmin(distances))
                dominant_color = centers[dominant_idx]
    except Exception:
        dominant_color = base_color

    diff = np.linalg.norm(rgb - dominant_color, axis=-1)

    diff_values = diff[mask]
    if diff_values.size == 0:
        return overlay

    diff_min = float(diff_values.min())
    diff_max = float(diff_values.max())
    diff_range = max(diff_max - diff_min, 1e-4)

    normalized_diff = np.clip((diff - diff_min) / diff_range, 0.0, 1.0)

    if diff_gamma <= 0.0:
        diff_gamma = 1.0
    if not np.isclose(diff_gamma, 1.0):
        normalized_diff = np.power(normalized_diff, diff_gamma)

    if detail_strength:
        tiered_weights = sorted(detail_strength, key=lambda item: item[0])
        weight_map = np.zeros_like(normalized_diff, dtype=np.float32)
        prev_threshold = 0.0
        for threshold, weight in tiered_weights:
            threshold_clamped = float(np.clip(threshold, 0.0, 1.0))
            threshold_clamped = max(threshold_clamped, prev_threshold)
            mask_band = (normalized_diff >= prev_threshold) & (
                normalized_diff < threshold_clamped
            )
            weight_map[mask_band] = float(weight)
            prev_threshold = threshold_clamped

        weight_map[normalized_diff >= prev_threshold] = float(tiered_weights[-1][1])
    else:
        weight_map = normalized_diff.astype(np.float32)

    base_alpha_floor = float(np.clip(base_alpha_floor, 0.0, 1.0))
    base_alpha_scale = float(np.clip(base_alpha_scale, 0.0, 1.0))
    if base_alpha_floor + base_alpha_scale > 1.0:
        base_alpha_scale = max(0.0, 1.0 - base_alpha_floor)

    detail_weight = base_alpha_floor + base_alpha_scale * weight_map

    if blend_mode == "softlight":
        t = np.clip(float(softlight_strength), 0.0, 1.0)
        detail_weight = np.clip(t * (1.0 - (1.0 - detail_weight) * (1.0 - detail_weight)) + (1.0 - t) * detail_weight, 0.0, 1.0)

    detail_alpha = alpha * detail_weight

    luminance = (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ) / 255.0
    luminance_values = luminance[mask]

    if luminance_values.size > 0 and 0.0 <= highlight_percentile <= 100.0:
        highlight_threshold = float(
            np.percentile(luminance_values, np.clip(highlight_percentile, 0.0, 100.0))
        )
        highlight_threshold = max(highlight_threshold, min_highlight_luminance)
        if highlight_scale < 0.0:
            highlight_scale = 0.0
        if highlight_scale > 1.0:
            highlight_scale = 1.0

        highlight_mask = (luminance >= highlight_threshold).astype(np.float32)
        highlight_mask *= mask.astype(np.float32)

        if np.any(highlight_mask > 1e-3):
            if highlight_blur_sigma > 0.0:
                radius = max(1, int(round(highlight_blur_sigma * 3)))
                ksize = radius * 2 + 1
                highlight_weight = cv2.GaussianBlur(
                    highlight_mask,
                    (ksize, ksize),
                    highlight_blur_sigma,
                )
            else:
                highlight_weight = highlight_mask

            highlight_weight = np.clip(highlight_weight, 0.0, 1.0)
            reduction = 1.0 - highlight_scale
            detail_alpha *= 1.0 - reduction * highlight_weight

    if not np.any(detail_alpha > 1e-3):
        return overlay

    rgba[..., 3] = np.clip(detail_alpha * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")

