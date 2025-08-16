from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Dict, Any
import os
import uuid
from retinaface import RetinaFace
from PIL import Image
import base64
# Add imports for MediaPipe and numpy
import numpy as np
import mediapipe as mp
import cv2  # Add this import for drawing overlays
from io import BytesIO
from ai_models.unet.inference_unet import load_model, predict_mask, colorize_mask, PALETTE
from ai_models.unet.inference_celeba_unet import process_image_with_celeba_unet
import torch

router = APIRouter()

@router.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """
    Detect faces in uploaded image using RetinaFace
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    # Save file tạm thời
    temp_filename = f"temp_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join("/tmp", temp_filename)
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Detect faces
        results = RetinaFace.detect_faces(temp_path)
        faces = []
        for face_id, face in results.items():
            box = face["facial_area"]
            faces.append({
                "face_id": face_id,
                "bounding_box": [int(x) for x in box],
                "landmarks": {k: [float(v[0]), float(v[1])] for k, v in face.get("landmarks", {}).items()}
            })
        # Đọc kích thước ảnh
        with Image.open(temp_path) as img:
            width, height = img.size
        return {
            "num_faces": int(len(faces)),
            "faces": faces,
            "image_size": {"width": int(width), "height": int(height)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/crop")
async def crop_face(
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...)
):
    """
    Crop face from image using bounding box, return base64 image
    """
    temp_filename = f"crop_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join("/tmp", temp_filename)
    crop_path = temp_path.replace(".jpg", "_face.jpg")
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    try:
        with Image.open(temp_path) as img:
            cropped = img.crop((x1, y1, x2, y2))
            cropped.save(crop_path)
            with open(crop_path, "rb") as cf:
                crop_b64 = base64.b64encode(cf.read()).decode()
        return {"cropped_image_base64": crop_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(crop_path):
            os.remove(crop_path) 

# Helper to draw overlays for each region

def draw_regions_on_image(image: np.ndarray, regions: dict) -> np.ndarray:
    overlay = image.copy()
    color_map = {
        "lips": (255, 0, 0),
        "left_eye": (0, 255, 0),
        "right_eye": (0, 255, 0),
        "left_eyebrow": (0, 0, 255),
        "right_eyebrow": (0, 0, 255),
        "left_cheek": (255, 255, 0),
        "right_cheek": (255, 255, 0),
        "contour": (255, 0, 255),
    }
    alpha = 0.4  # Transparency
    for region_name, points in regions.items():
        if not points:
            continue
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color_map.get(region_name, (255, 255, 255)))
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def image_to_base64(image: np.ndarray) -> str:
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@router.post("/makeup/extract")
async def extract_makeup(file: UploadFile = File(...)):
    """
    Extract makeup attributes (lips, eyes, eyebrows, blush, contour) from a cropped face image using MediaPipe Face Mesh
    Returns both the attributes and an annotated image with overlays for each region.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    temp_filename = f"makeup_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join("/tmp", temp_filename)
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    try:
        with Image.open(temp_path) as img:
            img = img.convert("RGB")
            img_np = np.array(img)
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(img_np)
            if not results.multi_face_landmarks:
                raise HTTPException(status_code=404, detail="No face landmarks detected.")
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = img_np.shape
            # Get landmark points as pixel coordinates
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
            # Define indices for each region (MediaPipe Face Mesh indices)
            LIPS_IDX = list(range(61, 88)) + list(range(291, 318))
            LEFT_EYE_IDX = list(range(33, 42)) + list(range(133, 144))
            RIGHT_EYE_IDX = list(range(263, 272)) + list(range(362, 373))
            LEFT_EYEBROW_IDX = list(range(46, 66))
            RIGHT_EYEBROW_IDX = list(range(276, 296))
            LEFT_CHEEK_IDX = list(range(205, 218))
            RIGHT_CHEEK_IDX = list(range(425, 438))
            JAWLINE_IDX = list(range(0, 17)) + list(range(267, 285))
            # Helper to get mask for a region
            def region_mask(indices):
                mask = np.zeros((h, w), dtype=np.uint8)
                region = np.array([points[i] for i in indices], dtype=np.int32)
                cv2.fillPoly(mask, [region], 1)
                return mask.astype(bool)
            # Helper to get average color
            def avg_color(mask):
                region_pixels = img_np[mask]
                if len(region_pixels) == 0:
                    return [0, 0, 0]
                return [int(np.mean(region_pixels[:, i])) for i in range(3)]
            # Masks and colors
            lips_mask = region_mask(LIPS_IDX)
            left_eye_mask = region_mask(LEFT_EYE_IDX)
            right_eye_mask = region_mask(RIGHT_EYE_IDX)
            left_eyebrow_mask = region_mask(LEFT_EYEBROW_IDX)
            right_eyebrow_mask = region_mask(RIGHT_EYEBROW_IDX)
            left_cheek_mask = region_mask(LEFT_CHEEK_IDX)
            right_cheek_mask = region_mask(RIGHT_CHEEK_IDX)
            # Colors
            lips_color = avg_color(lips_mask)
            left_eye_color = avg_color(left_eye_mask)
            right_eye_color = avg_color(right_eye_mask)
            left_eyebrow_color = avg_color(left_eyebrow_mask)
            right_eyebrow_color = avg_color(right_eyebrow_mask)
            left_cheek_color = avg_color(left_cheek_mask)
            right_cheek_color = avg_color(right_cheek_mask)
            # Contour (jawline) shape: return as list of points
            contour_points = [points[i] for i in JAWLINE_IDX]
            # Prepare regions for overlay
            regions = {
                "lips": [points[i] for i in LIPS_IDX],
                "left_eye": [points[i] for i in LEFT_EYE_IDX],
                "right_eye": [points[i] for i in RIGHT_EYE_IDX],
                "left_eyebrow": [points[i] for i in LEFT_EYEBROW_IDX],
                "right_eyebrow": [points[i] for i in RIGHT_EYEBROW_IDX],
                "left_cheek": [points[i] for i in LEFT_CHEEK_IDX],
                "right_cheek": [points[i] for i in RIGHT_CHEEK_IDX],
                "contour": contour_points,
            }
            # Draw overlays
            annotated_img = draw_regions_on_image(img_np.copy(), regions)
            annotated_img_b64 = image_to_base64(annotated_img)
            return {
                "lips_color": lips_color,
                "left_eye_color": left_eye_color,
                "right_eye_color": right_eye_color,
                "left_eyebrow_color": left_eyebrow_color,
                "right_eyebrow_color": right_eyebrow_color,
                "left_cheek_color": left_cheek_color,
                "right_cheek_color": right_cheek_color,
                "contour_shape": contour_points,
                "annotated_image": annotated_img_b64
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Makeup extraction failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path) 

@router.post("/makeup/unet_extract")
async def unet_extract_makeup(file: UploadFile = File(...)):
    """
    Extract face regions using U-Net, return colorized mask and average color for each region.
    """
    temp_filename = f"unet_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join("/tmp", temp_filename)
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    try:
        pil_img = Image.open(temp_path).convert('RGB')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(device)
        mask = predict_mask(model, pil_img, device)
        color_mask_img = colorize_mask(mask)
        # Encode color mask as base64
        buffered = BytesIO()
        color_mask_img.save(buffered, format="PNG")
        mask_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # Compute average color for each region
        region_colors = {}
        np_img = np.array(pil_img.resize(mask.shape[::-1]))
        for idx, name in enumerate([
            "background", "skin", "lips", "eyes", "eyebrows", "cheeks", "other"
        ]):
            region_pixels = np_img[mask == idx]
            if len(region_pixels) == 0:
                region_colors[name] = [0, 0, 0]
            else:
                region_colors[name] = [int(np.mean(region_pixels[:, i])) for i in range(3)]
        return {
            "colorized_mask": f"data:image/png;base64,{mask_b64}",
            "region_colors": region_colors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"U-Net extraction failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path) 

@router.post("/makeup/celeba_unet_extract")
async def celeba_unet_extract_makeup(file: UploadFile = File(...)):
    """
    Extract makeup attributes using CelebAMask-HQ U-Net model
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    # Save file temporarily
    temp_filename = f"celeba_unet_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join("/tmp", temp_filename)
    
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Process with CelebAMask-HQ U-Net
        result = process_image_with_celeba_unet(temp_path)
        
        return {
            "colorized_mask": f"data:image/png;base64,{result['colorized_mask']}",
            "annotated_image": f"data:image/png;base64,{result['annotated_image']}",
            "region_colors": result["region_colors"],
            "attributes": result["attributes"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CelebAMask-HQ U-Net extraction failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path) 