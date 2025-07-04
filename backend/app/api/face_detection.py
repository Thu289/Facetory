from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

router = APIRouter()

@router.post("/detect")
async def detect_faces():
    """
    Detect faces in uploaded image (placeholder for Phase 1.3)
    """
    return {
        "message": "Face detection endpoint - coming in Phase 1.3",
        "faces_detected": 0,
        "bounding_boxes": []
    }

@router.post("/crop")
async def crop_face():
    """
    Crop face from image (placeholder for Phase 1.4)
    """
    return {
        "message": "Face crop endpoint - coming in Phase 1.4",
        "cropped_image_path": None
    } 