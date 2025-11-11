"""
Thumbnail Generation Service
Creates preview thumbnails for makeup styles
"""

import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any
import base64
from io import BytesIO


def generate_thumbnail(
    original_image: np.ndarray,
    colorized_mask: np.ndarray,
    annotated_image: np.ndarray,
    size: tuple = (256, 256)
) -> str:
    """
    Generate a thumbnail preview showing original, mask, and result
    
    Args:
        original_image: Original cropped face image
        colorized_mask: Colorized segmentation mask
        annotated_image: Image with mask overlay
        size: Thumbnail size (width, height)
    
    Returns:
        Base64 encoded thumbnail image
    """
    # Resize all images to thumbnail size
    orig_resized = cv2.resize(original_image, size)
    mask_resized = cv2.resize(colorized_mask, size)
    annotated_resized = cv2.resize(annotated_image, size)
    
    # Create a 3-panel thumbnail: original | mask | result
    thumbnail = np.hstack([orig_resized, mask_resized, annotated_resized])
    
    # Convert to PIL Image
    thumbnail_pil = Image.fromarray(thumbnail)
    
    # Convert to base64
    buffer = BytesIO()
    thumbnail_pil.save(buffer, format="PNG")
    thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{thumbnail_b64}"

