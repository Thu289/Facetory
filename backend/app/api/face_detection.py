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
from ai_models.BiseNet.inference_bisenet import process_image_bisenet, CELEBA_ATTRIBUTES
from app.services.style_extraction import extract_makeup_style
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

@router.post("/makeup/style_extract")
async def extract_makeup_style_api(file: UploadFile = File(...)):
    """
    Extract makeup style from uploaded image following the flow in makeup_filter_system.md:
    
    Step 1: RetinaFace → Detect face region
    Step 2: Crop face
    Step 3: BiSeNet → Segment facial regions (19 attributes)
    Step 4: Style Extraction → LAB color space, K-means, histogram analysis
    Step 5: Generate style parameters (JSON format)
    
    Returns comprehensive style data including:
    - Segmentation mask and colorized visualization
    - Style parameters for lips, eyes, eyebrows, skin
    - LAB color space analysis
    - Coverage intensity, blend softness, texture type
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    # Save file temporarily
    temp_filename = f"style_extract_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join("/tmp", temp_filename)
    
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Step 1: RetinaFace - Detect face
        print("🔍 Step 1: Detecting face with RetinaFace...")
        face_results = RetinaFace.detect_faces(temp_path)
        
        if not face_results:
            raise HTTPException(status_code=404, detail="No faces detected in the image.")
        
        # Get first face bounding box
        first_face = list(face_results.values())[0]
        face_box = first_face["facial_area"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = [int(coord) for coord in face_box]
        
        # Step 2: Crop face
        print("✂️  Step 2: Cropping face region...")
        with Image.open(temp_path) as img:
            # Add padding around face (20% on each side)
            padding = int((x2 - x1) * 0.2)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.width, x2 + padding)
            y2 = min(img.height, y2 + padding)
            
            cropped_face = img.crop((x1, y1, x2, y2))
            
            # Save cropped face temporarily
            cropped_path = temp_path.replace(".jpg", "_cropped.jpg")
            cropped_face.save(cropped_path)
        
        # Step 3: BiSeNet - Segment facial regions
        print("🎨 Step 3: Segmenting face with BiSeNet...")
        bisenet_result = process_image_bisenet(
            cropped_path, 
            device=device,
            return_mask_array=True
        )
        
        if bisenet_result is None:
            raise HTTPException(
                status_code=500, 
                detail="BiSeNet segmentation failed. Please ensure BiSeNet is properly set up (run setup_bisenet.sh)."
            )
        
        segmentation_mask = bisenet_result['mask']
        colorized_mask = bisenet_result['colorized_mask']
        attribute_mapping = bisenet_result['attribute_mapping']
        
        # Convert PIL image to numpy for style extraction
        import numpy as np
        image_rgb = np.array(cropped_face.convert('RGB'))
        
        # Extract average color for each region (for visualization)
        # IMPORTANT: Use ORIGINAL image (image_rgb), NOT the annotated/blended image
        print("🎨 Extracting region colors from ORIGINAL image...")
        region_colors = {}
        region_color_images = {}
        
        # Use the ORIGINAL cropped face image (image_rgb) for color extraction
        # Resize segmentation_mask to match original image size if needed
        if segmentation_mask.shape[:2] != image_rgb.shape[:2]:
            # Resize mask to match image (use nearest neighbor to preserve class IDs)
            mask_resized = cv2.resize(
                segmentation_mask.astype(np.uint8),
                (image_rgb.shape[1], image_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(segmentation_mask.dtype)
        else:
            mask_resized = segmentation_mask
        
        # Extract color for each attribute from ORIGINAL image
        for class_id, attr_name in attribute_mapping.items():
            if class_id == 0:  # Skip background
                continue
            region_mask = (mask_resized == class_id)
            if np.any(region_mask):
                # Extract pixels from ORIGINAL image_rgb
                region_pixels = image_rgb[region_mask]
                if len(region_pixels) > 0:
                    avg_color = np.mean(region_pixels, axis=0).astype(int).tolist()
                    region_colors[attr_name] = {
                        "rgb": avg_color,
                        "hex": f"#{avg_color[0]:02x}{avg_color[1]:02x}{avg_color[2]:02x}",
                        "pixel_count": int(np.sum(region_mask))
                    }
                    
                    # Create a small preview image (50x50) showing the average color
                    color_preview = np.full((50, 50, 3), avg_color, dtype=np.uint8)
                    color_preview_pil = Image.fromarray(color_preview)
                    color_buffer = BytesIO()
                    color_preview_pil.save(color_buffer, format="PNG")
                    color_preview_b64 = base64.b64encode(color_buffer.getvalue()).decode("utf-8")
                    region_color_images[attr_name] = f"data:image/png;base64,{color_preview_b64}"
        
        # Step 4: Style Extraction - LAB color space, K-means, histogram
        print("🔬 Step 4: Extracting style parameters...")
        style_data = extract_makeup_style(
            image_rgb=image_rgb,
            segmentation_mask=segmentation_mask,
            attribute_mapping=attribute_mapping
        )
        
        # Step 5: Prepare response with visualizations
        # Convert colorized mask to base64
        colorized_pil = Image.fromarray(colorized_mask)
        colorized_buffer = BytesIO()
        colorized_pil.save(colorized_buffer, format="PNG")
        colorized_b64 = base64.b64encode(colorized_buffer.getvalue()).decode("utf-8")
        
        # Create annotated image (overlay mask on original)
        # Resize colorized_mask to match image_rgb size
        if colorized_mask.shape[:2] != image_rgb.shape[:2]:
            colorized_mask_resized = cv2.resize(
                colorized_mask, 
                (image_rgb.shape[1], image_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            colorized_mask_resized = colorized_mask
        
        annotated_image = image_rgb.copy()
        alpha = 0.4
        annotated_image = cv2.addWeighted(annotated_image, 1-alpha, colorized_mask_resized, alpha, 0)
        annotated_pil = Image.fromarray(annotated_image)
        annotated_buffer = BytesIO()
        annotated_pil.save(annotated_buffer, format="PNG")
        annotated_b64 = base64.b64encode(annotated_buffer.getvalue()).decode("utf-8")
        
        # Also include cropped face image for comparison
        cropped_pil = cropped_face.copy()
        cropped_buffer = BytesIO()
        cropped_pil.save(cropped_buffer, format="PNG")
        cropped_b64 = base64.b64encode(cropped_buffer.getvalue()).decode("utf-8")
        
        return {
            "success": True,
            "style_id": style_data.get('style_id'),
            "style_parameters": {
                "lips": style_data.get('lips', {}),
                "eyes": style_data.get('eyes', {}),
                "eyebrows": style_data.get('eyebrows', {}),
                "skin": style_data.get('skin', {})
            },
            "segmentation": {
                "colorized_mask": f"data:image/png;base64,{colorized_b64}",
                "annotated_image": f"data:image/png;base64,{annotated_b64}",
                "original_cropped": f"data:image/png;base64,{cropped_b64}",
                "attributes": bisenet_result['attributes'],
                "attribute_mapping": attribute_mapping,
                "region_colors": region_colors,  # RGB và hex colors cho mỗi vùng
                "region_color_previews": region_color_images  # Preview images cho mỗi vùng
            },
            "face_detection": {
                "bounding_box": [x1, y1, x2, y2],
                "original_size": {"width": cropped_face.width, "height": cropped_face.height}
            },
            "processing_info": {
                "device": device,
                "model": "BiSeNet (CelebAMask-HQ)",
                "segmentation_classes": len(CELEBA_ATTRIBUTES) + 1
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Style extraction failed: {str(e)}"
        )
    finally:
        # Cleanup temporary files
        for path in [temp_path, temp_path.replace(".jpg", "_cropped.jpg")]:
            if os.path.exists(path):
                os.remove(path) 