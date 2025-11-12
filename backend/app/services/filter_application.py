"""Filter Application Service (RGBA overlays)."""

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.services.storage import MinioService
from app.services.facemesh_mesh import extract_normalized_landmarks


def _load_overlay_average(mask_url: str, minio_service: MinioService) -> Optional[List[int]]:
    """Load stored RGBA overlay and compute average RGB where alpha > 0."""
    if not mask_url:
        return None

    object_name = mask_url
    if mask_url.startswith("/api/"):
        from urllib.parse import unquote

        object_name = mask_url.replace("/api/makeup/storage/file/", "")
        object_name = unquote(object_name)

    temp_png_path = tempfile.mktemp(suffix=".png")
    try:
        minio_service.client.fget_object(
            minio_service.bucket_name,
            object_name,
            temp_png_path,
        )
        overlay = Image.open(temp_png_path).convert("RGBA")
        overlay_np = np.array(overlay)
        alpha = overlay_np[..., 3].astype(np.float32) / 255.0
        mask = alpha > 0.1
        if not np.any(mask):
            return None
        rgb_values = overlay_np[..., :3][mask]
        avg = rgb_values.mean(axis=0).astype(int).tolist()
        return avg
    except Exception:
        return None
    finally:
        if os.path.exists(temp_png_path):
            os.remove(temp_png_path)


def _load_overlay_image(mask_url: str, minio_service: MinioService) -> Optional[np.ndarray]:
    """
    Load stored RGBA overlay image as numpy array.

    Returns:
        NumPy array (H, W, 4) in uint8 or None on failure.
    """
    if not mask_url:
        return None

    object_name = mask_url
    if mask_url.startswith("/api/"):
        from urllib.parse import unquote

        object_name = mask_url.replace("/api/makeup/storage/file/", "")
        object_name = unquote(object_name)

    temp_png_path = tempfile.mktemp(suffix=".png")
    try:
        minio_service.client.fget_object(
            minio_service.bucket_name,
            object_name,
            temp_png_path,
        )
        overlay = Image.open(temp_png_path).convert("RGBA")
        return np.array(overlay)
    except Exception:
        return None
    finally:
        if os.path.exists(temp_png_path):
            os.remove(temp_png_path)


MESH_REGION_FALLBACKS: Dict[str, str] = {
    "lips_upper": "lips",
    "lips_lower": "lips",
    "eyebrow_left": "eyebrows",
    "eyebrow_right": "eyebrows",
}


def _resolve_region_mesh(region_meshes: Optional[Dict[str, Dict[str, object]]], region: str) -> Optional[Dict[str, object]]:
    if not region_meshes:
        return None
    if region in region_meshes:
        return region_meshes[region]
    fallback = MESH_REGION_FALLBACKS.get(region)
    if fallback and fallback in region_meshes:
        return region_meshes[fallback]
    return None


def _warp_overlay_with_mesh(
    overlay_rgba: np.ndarray,
    mesh_data: Dict[str, object],
    target_landmarks: Dict[int, Tuple[float, float]],
    target_size: Tuple[int, int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Warp overlay image onto the target face using stored mesh data.
    Returns warped RGB (float32) and alpha (float32 in [0, 1]).
    """
    points_data = mesh_data.get("points")
    triangles = mesh_data.get("triangles")
    if not isinstance(points_data, list) or not isinstance(triangles, list):
        return None

    target_height, target_width = target_size
    overlay_height, overlay_width = overlay_rgba.shape[:2]

    base_points: List[Tuple[float, float]] = []
    target_points: List[Tuple[float, float]] = []

    for point in points_data:
        if not isinstance(point, dict):
            continue
        idx = point.get("index")
        x_norm = point.get("x")
        y_norm = point.get("y")
        if idx is None or x_norm is None or y_norm is None:
            continue
        idx = int(idx)
        if idx not in target_landmarks:
            return None

        base_points.append((float(x_norm) * overlay_width, float(y_norm) * overlay_height))
        target_x_norm, target_y_norm = target_landmarks[idx]
        target_points.append((target_x_norm * target_width, target_y_norm * target_height))

    if len(base_points) < 3 or len(triangles) == 0:
        return None

    overlay_rgb = overlay_rgba[..., :3].astype(np.float32)
    overlay_alpha = overlay_rgba[..., 3].astype(np.float32) / 255.0

    warped_rgb = np.zeros((target_height, target_width, 3), dtype=np.float32)
    warped_alpha = np.zeros((target_height, target_width), dtype=np.float32)

    base_array = np.array(base_points, dtype=np.float32)
    target_array = np.array(target_points, dtype=np.float32)

    for tri in triangles:
        if not isinstance(tri, (list, tuple)) or len(tri) != 3:
            continue
        i1, i2, i3 = map(int, tri)
        src_tri = base_array[[i1, i2, i3]]
        dst_tri = target_array[[i1, i2, i3]]

        if cv2.contourArea(src_tri) < 1e-4 or cv2.contourArea(dst_tri) < 1e-4:
            continue

        src_rect = cv2.boundingRect(src_tri)
        dst_rect = cv2.boundingRect(dst_tri)

        if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
            continue

        x_src, y_src, w_src, h_src = src_rect
        x_dst, y_dst, w_dst, h_dst = dst_rect

        src_crop_rgb = overlay_rgb[y_src : y_src + h_src, x_src : x_src + w_src]
        src_crop_alpha = overlay_alpha[y_src : y_src + h_src, x_src : x_src + w_src]

        src_tri_rect = src_tri - np.array([x_src, y_src], dtype=np.float32)
        dst_tri_rect = dst_tri - np.array([x_dst, y_dst], dtype=np.float32)

        warp_mat = cv2.getAffineTransform(src_tri_rect, dst_tri_rect)

        warped_rgb_roi = cv2.warpAffine(
            src_crop_rgb,
            warp_mat,
            (w_dst, h_dst),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        warped_alpha_roi = cv2.warpAffine(
            src_crop_alpha,
            warp_mat,
            (w_dst, h_dst),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros((h_dst, w_dst), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_rect), 1.0)
        warped_rgb_roi *= mask[..., None]
        warped_alpha_roi *= mask

        roi_rgb = warped_rgb[y_dst : y_dst + h_dst, x_dst : x_dst + w_dst]
        roi_alpha = warped_alpha[y_dst : y_dst + h_dst, x_dst : x_dst + w_dst]

        prev_alpha = roi_alpha
        additional_alpha = warped_alpha_roi * (1.0 - prev_alpha)
        new_alpha = prev_alpha + additional_alpha

        blend_mask = new_alpha > 1e-6
        if not np.any(blend_mask):
            continue

        roi_rgb[blend_mask] = (
            roi_rgb[blend_mask] * prev_alpha[blend_mask, None]
            + warped_rgb_roi[blend_mask] * warped_alpha_roi[blend_mask, None] * (1.0 - prev_alpha[blend_mask, None])
        ) / new_alpha[blend_mask, None]
        roi_alpha[blend_mask] = new_alpha[blend_mask]

        warped_rgb[y_dst : y_dst + h_dst, x_dst : x_dst + w_dst] = roi_rgb
        warped_alpha[y_dst : y_dst + h_dst, x_dst : x_dst + w_dst] = roi_alpha

    return warped_rgb, np.clip(warped_alpha, 0.0, 1.0)


def apply_style_to_image(
    image: np.ndarray,
    style_data: Dict[str, Any],
    intensity: float = 1.0,
    use_regions: bool = False,
    face_mask: Optional[np.ndarray] = None,
    region_masks: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """
    Apply makeup style to image using RGBA masks (average color blending).
    
    Args:
        image: RGB image array (H, W, 3)
        style_data: Style data with download_urls and style_parameters
        intensity: Overall filter intensity
        use_regions: Whether to apply region-specific filters (requires segmentation)
    
    Returns:
        Filtered image array
    """
    from app.services.storage import MinioService
    
    mask_urls = style_data.get('download_urls', {}).get('region_masks', {})
    minio_service = MinioService() if mask_urls else None

    result = image.astype(np.float32).copy()
    
    # Create face mask if not provided (apply filter to entire image)
    if face_mask is None:
        # If no mask provided, create one that covers entire image
        face_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
    else:
        # Ensure mask matches image size
        if face_mask.shape[:2] != image.shape[:2]:
            face_mask = cv2.resize(face_mask.astype(np.float32), 
                                  (image.shape[1], image.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
    
    # Normalize mask to [0, 1]
    if face_mask.max() > 1.0:
        face_mask = face_mask.astype(np.float32) / 255.0
    
    if not use_regions or not region_masks:
        return result.astype(np.uint8)

    # Build lookup for style parameters
    style_params = style_data.get("style_parameters", {})
    # Legacy support: allow top-level keys (e.g., style_data['lips'])
    for key in [
        "lips",
        "lips_upper",
        "lips_lower",
        "eyebrows",
        "eyebrow_left",
        "eyebrow_right",
        "nose",
        "skin",
        "cheeks",
        "eyes",
    ]:
        if key not in style_params and key in style_data and isinstance(style_data[key], dict):
            style_params[key] = style_data[key]

    metadata = style_data.get("metadata") or {}
    raw_region_meshes = metadata.get("region_meshes") or style_data.get("region_meshes") or {}
    region_meshes = raw_region_meshes if isinstance(raw_region_meshes, dict) else {}

    target_landmarks: Optional[Dict[int, Tuple[float, float]]] = None
    if region_meshes:
        try:
            target_landmarks = extract_normalized_landmarks(np.ascontiguousarray(image))
        except Exception:
            target_landmarks = None
        if target_landmarks is None and region_meshes:
            print("[filter_application] Warning: face landmarks unavailable; falling back to mask-based blending.")

    overlay_cache: Dict[str, Optional[np.ndarray]] = {}
    region_mask_map = region_masks or {}

    region_keys: List[str] = []
    region_keys.extend(region_mask_map.keys())
    region_keys.extend(style_params.keys())
    region_keys.extend(mask_urls.keys())
    region_keys = list(dict.fromkeys(region_keys))

    for region in region_keys:
        if region not in style_params:
            continue

        if region == 'lips':
            has_split_mask = any(key in region_mask_map for key in ('lips_upper', 'lips_lower'))
            has_split_params = any(key in style_params for key in ('lips_upper', 'lips_lower'))
            if has_split_mask and has_split_params:
                continue
        if region == 'eyebrows':
            has_split_mask = any(key in region_mask_map for key in ('eyebrow_left', 'eyebrow_right'))
            has_split_params = any(key in style_params for key in ('eyebrow_left', 'eyebrow_right'))
            if has_split_mask and has_split_params:
                continue

        params = style_params.get(region, {})
        if not isinstance(params, dict):
            continue

        average_rgb = params.get("average_rgb")
        if average_rgb is None and mask_urls.get(region) and minio_service:
            try:
                overlay_color = _load_overlay_average(mask_urls[region], minio_service)
                if overlay_color is not None:
                    average_rgb = overlay_color
            except Exception:
                average_rgb = None

        if average_rgb is None and not mask_urls.get(region):
            continue

        coverage_value = params.get("coverage_intensity", 1.0)
        try:
            coverage_float = float(np.clip(coverage_value, 0.0, 1.0))
        except Exception:
            coverage_float = 1.0
        coverage_float *= float(np.clip(intensity, 0.0, 1.0))

        mask: Optional[np.ndarray] = None
        region_mask = region_mask_map.get(region)
        if region_mask is not None:
            mask = region_mask.astype(np.float32)
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask = np.clip(mask, 0.0, 1.0)
            if not np.any(mask > 1e-3):
                mask = None

        overlay_key = mask_urls.get(region)
        overlay_rgba: Optional[np.ndarray] = None
        if overlay_key and minio_service:
            if overlay_key not in overlay_cache:
                overlay_cache[overlay_key] = _load_overlay_image(overlay_key, minio_service)
            overlay_rgba = overlay_cache[overlay_key]

        aligned_rgb: Optional[np.ndarray] = None
        aligned_alpha: Optional[np.ndarray] = None

        if overlay_rgba is not None:
            mesh_entry = _resolve_region_mesh(region_meshes, region)
            if mesh_entry and target_landmarks is not None:
                warped = _warp_overlay_with_mesh(
                    overlay_rgba,
                    mesh_entry,
                    target_landmarks,
                    (image.shape[0], image.shape[1]),
                )
                if warped is not None:
                    aligned_rgb, aligned_alpha = warped

            if aligned_rgb is None or aligned_alpha is None:
                overlay_rgb_full = overlay_rgba[..., :3].astype(np.float32)
                overlay_alpha_full = overlay_rgba[..., 3].astype(np.float32) / 255.0
                overlay_alpha_full = np.clip(overlay_alpha_full, 0.0, 1.0)

                if mask is not None:
                    overlay_mask_bool = overlay_alpha_full > 1e-3
                    target_mask_bool = mask > 1e-3

                    if np.any(overlay_mask_bool) and np.any(target_mask_bool):
                        overlay_coords = np.argwhere(overlay_mask_bool)
                        target_coords = np.argwhere(target_mask_bool)

                        overlay_y0 = int(overlay_coords[:, 0].min())
                        overlay_y1 = int(overlay_coords[:, 0].max()) + 1
                        overlay_x0 = int(overlay_coords[:, 1].min())
                        overlay_x1 = int(overlay_coords[:, 1].max()) + 1

                        target_y0 = int(target_coords[:, 0].min())
                        target_y1 = int(target_coords[:, 0].max()) + 1
                        target_x0 = int(target_coords[:, 1].min())
                        target_x1 = int(target_coords[:, 1].max()) + 1

                        overlay_h = max(0, overlay_y1 - overlay_y0)
                        overlay_w = max(0, overlay_x1 - overlay_x0)
                        target_h = max(0, target_y1 - target_y0)
                        target_w = max(0, target_x1 - target_x0)

                        if overlay_h > 1 and overlay_w > 1 and target_h > 1 and target_w > 1:
                            overlay_rgb_crop = overlay_rgb_full[overlay_y0:overlay_y1, overlay_x0:overlay_x1]
                            overlay_alpha_crop = overlay_alpha_full[overlay_y0:overlay_y1, overlay_x0:overlay_x1]

                            resized_rgb = cv2.resize(
                                overlay_rgb_crop,
                                (target_w, target_h),
                                interpolation=cv2.INTER_LINEAR,
                            ).astype(np.float32)
                            resized_alpha = cv2.resize(
                                overlay_alpha_crop,
                                (target_w, target_h),
                                interpolation=cv2.INTER_LINEAR,
                            ).astype(np.float32)

                            target_mask_crop = mask[target_y0:target_y1, target_x0:target_x1]
                            resized_alpha = np.clip(resized_alpha * target_mask_crop, 0.0, 1.0)

                            aligned_rgb = np.zeros_like(result, dtype=np.float32)
                            aligned_alpha = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                            aligned_rgb[target_y0:target_y1, target_x0:target_x1] = resized_rgb
                            aligned_alpha[target_y0:target_y1, target_x0:target_x1] = resized_alpha

                if aligned_rgb is None or aligned_alpha is None:
                    if overlay_rgba.shape[0] != image.shape[0] or overlay_rgba.shape[1] != image.shape[1]:
                        overlay_rgb_full = cv2.resize(
                            overlay_rgba[..., :3].astype(np.float32),
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        overlay_alpha_full = cv2.resize(
                            overlay_rgba[..., 3].astype(np.float32) / 255.0,
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        overlay_rgb_full = overlay_rgba[..., :3].astype(np.float32)
                        overlay_alpha_full = overlay_rgba[..., 3].astype(np.float32) / 255.0
                    aligned_rgb = overlay_rgb_full
                    aligned_alpha = np.clip(overlay_alpha_full, 0.0, 1.0)

            if aligned_rgb is not None and aligned_alpha is not None:
                combined_mask = np.clip(aligned_alpha, 0.0, 1.0) * coverage_float
                if mask is not None:
                    combined_mask *= mask
                combined_mask = np.clip(combined_mask, 0.0, 1.0)
                combined_mask *= face_mask

                if np.any(combined_mask > 0):
                    mask_3d = np.stack([combined_mask, combined_mask, combined_mask], axis=-1)
                    if region == "nose":
                        base_norm = np.clip(result / 255.0, 0.0, 1.0)
                        overlay_norm = np.clip(aligned_rgb / 255.0, 0.0, 1.0)
                        softlight = np.where(
                            overlay_norm <= 0.5,
                            base_norm - (1 - 2 * overlay_norm) * base_norm * (1 - base_norm),
                            base_norm + (2 * overlay_norm - 1) * (np.sqrt(base_norm) - base_norm),
                        )
                        softlight = np.clip(softlight * 255.0, 0.0, 255.0)
                        result = result * (1 - mask_3d) + softlight * mask_3d
                    else:
                        result = result * (1 - mask_3d) + aligned_rgb * mask_3d
                continue

        if mask is None:
            continue

        color = np.array(average_rgb if average_rgb is not None else [0, 0, 0], dtype=np.float32)
        color_image = np.ones_like(result) * color
        mask_blend = mask * coverage_float
        mask_blend *= face_mask
        if not np.any(mask_blend > 0):
            continue
        mask_3d = np.stack([mask_blend, mask_blend, mask_blend], axis=-1)
        result = result * (1 - mask_3d) + color_image * mask_3d
    
    # Ensure result is clipped to valid range
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def apply_style_to_image_file(
    image_path: str,
    style_data: Dict[str, Any],
    intensity: float = 1.0,
    output_path: Optional[str] = None
) -> str:
    """
    Apply makeup style to image file
    
    Args:
        image_path: Path to input image
        style_data: Style data with download_urls
        intensity: Filter intensity
        output_path: Optional output path
    
    Returns:
        Path to filtered image
    """
    # Load image
    img = Image.open(image_path)
    img_rgb = np.array(img.convert('RGB'))
    
    # Apply filter
    filtered = apply_style_to_image(img_rgb, style_data, intensity)
    
    # Save result
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.jpg')
    
    filtered_img = Image.fromarray(filtered)
    filtered_img.save(output_path, quality=95)
    
    return output_path

