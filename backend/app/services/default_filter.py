"""Default Filter Generator using RGBA overlays."""

from typing import Any, Dict
import os
import tempfile
import numpy as np
import cv2
from PIL import Image

from app.services.storage import MinioService
from app.services.style_storage import save_style


def _create_lip_mask(height: int, width: int) -> np.ndarray:
    """Create a simple elliptical mask to mimic lips."""
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, int(height * 0.62))
    axes = (int(width * 0.22), int(height * 0.08))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return (mask / 255.0).astype(np.float32)


def create_default_red_lips_style() -> Dict[str, Any]:
    """Create a default style with red lips overlay for testing."""
    style_id = "default_red_lips"
    red_rgb = [189, 103, 101]

    overlay_h, overlay_w = 256, 256
    lip_mask = _create_lip_mask(overlay_h, overlay_w)

    rgba = np.zeros((overlay_h, overlay_w, 4), dtype=np.uint8)
    mask_bool = lip_mask > 0
    rgba[..., :3][mask_bool] = red_rgb
    rgba[..., 3][mask_bool] = (lip_mask[mask_bool] * 255).astype(np.uint8)

    overlay_image = Image.fromarray(rgba, mode="RGBA")

    temp_dir = tempfile.mkdtemp(prefix=f"default_style_{style_id}_")
    overlay_path = os.path.join(temp_dir, f"{style_id}_lips_mask.png")
    overlay_image.save(overlay_path)

    minio_service = MinioService()
    object_name = f"styles/{style_id}/region_masks/{os.path.basename(overlay_path)}"
    minio_service.client.fput_object(
        minio_service.bucket_name,
        object_name,
        overlay_path,
    )
    mask_url = minio_service.get_file_url(object_name, expires=86400 * 7, use_proxy=True)

    filter_preview = Image.new("RGB", (overlay_w, overlay_h), (0, 0, 0))
    preview_buffer = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        filter_preview.save(preview_buffer.name, format="PNG")
        with open(preview_buffer.name, "rb") as f:
            import base64

            preview_b64 = base64.b64encode(f.read()).decode("utf-8")
            preview_data_url = f"data:image/png;base64,{preview_b64}"
    finally:
        preview_buffer.close()
        os.remove(preview_buffer.name)

    style_data = {
        "style_id": style_id,
        "name": "Default Red Lips",
        "description": "Default filter with bright red lipstick for testing",
        "download_urls": {
            "region_masks": {"lips": mask_url},
            "style_parameters": "",
        },
        "style_parameters": {
            "lips": {
                "average_rgb": red_rgb,
                "coverage_intensity": 0.8,
            }
        },
        "storage_info": {
            "minio_bucket": minio_service.bucket_name,
            "storage_path": f"styles/{style_id}/",
            "database_stored": False,
        },
        "metadata": {
            "filter_preview": preview_data_url,
            "segmentation": {},
            "face_detection": {},
        },
        "created_at": "2024-01-01T00:00:00",
    }

    save_style(style_id, style_data)

    os.remove(overlay_path)
    os.rmdir(temp_dir)

    return style_data



