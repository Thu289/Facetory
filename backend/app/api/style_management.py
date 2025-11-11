from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional, Dict, Any
import os
import uuid
import json
import tempfile
import shutil
import base64
from datetime import datetime
from PIL import Image
from io import BytesIO
import numpy as np

from app.services.style_storage import save_style, get_style, list_styles
from app.services.storage import MinioService
from app.services.facemesh_mesh import compute_region_meshes
from app.core.config import settings

router = APIRouter()


@router.get("/storage/file/{object_name:path}")
async def proxy_storage_file(object_name: str):
    from urllib.parse import unquote
    from fastapi.responses import StreamingResponse
    from app.services.storage import MinioService
    import io
    
    try:
        object_name = unquote(object_name)
        
        minio_service = MinioService()
        from minio.error import S3Error
        try:
            data = minio_service.client.get_object(
                minio_service.bucket_name,
                object_name
            )
            
            content_type = "application/octet-stream"
            if object_name.endswith('.glsl'):
                content_type = "text/plain"
            elif object_name.endswith('.bin'):
                content_type = "application/octet-stream"
            elif object_name.endswith('.json'):
                content_type = "application/json"
            elif object_name.endswith(('.png', '.jpg', '.jpeg')):
                content_type = f"image/{object_name.split('.')[-1]}"
            def iterfile():
                while True:
                    chunk = data.read(8192)
                    if not chunk:
                        break
                    yield chunk
                data.close()
                data.release_conn()
            
            return StreamingResponse(
                iterfile(),
                media_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        except S3Error as e:
            raise HTTPException(status_code=404, detail=f"File not found: {object_name}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")


@router.post("/style/create_complete")
async def create_complete_style(
    file: UploadFile = File(...),
    preview_file: Optional[UploadFile] = File(None),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    store_in_db: bool = Form(False)
):
    from retinaface import RetinaFace
    import cv2
    import torch
    import numpy as np
    
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    temp_filename = f"style_complete_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join("/tmp", temp_filename)
    
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    preview_temp_path = None
    if preview_file is not None:
        if not preview_file.content_type or not preview_file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Preview file must be an image.")
        preview_temp_filename = f"style_preview_{uuid.uuid4().hex[:8]}.jpg"
        preview_temp_path = os.path.join("/tmp", preview_temp_filename)
        with open(preview_temp_path, "wb") as pf:
            preview_content = await preview_file.read()
            pf.write(preview_content)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        from ai_models.BiseNet.inference_bisenet import process_image_bisenet
        from app.services.style_extraction import extract_makeup_style
        
        face_results = RetinaFace.detect_faces(temp_path)
        if not face_results:
            raise HTTPException(status_code=404, detail="No faces detected in the image.")
        
        # Get first face and crop
        first_face = list(face_results.values())[0]
        face_box = first_face["facial_area"]
        x1, y1, x2, y2 = [int(coord) for coord in face_box]
        
        with Image.open(temp_path) as img:
            padding = int((x2 - x1) * 0.2)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.width, x2 + padding)
            y2 = min(img.height, y2 + padding)
            cropped_face = img.crop((x1, y1, x2, y2))
            cropped_path = temp_path.replace(".jpg", "_cropped.jpg")
            cropped_face.save(cropped_path)
        
        # Step 3: BiSeNet segmentation
        bisenet_result = process_image_bisenet(cropped_path, device=device, return_mask_array=True)
        if bisenet_result is None:
            raise HTTPException(status_code=500, detail="BiSeNet segmentation failed.")
        
        print("=" * 80)
        print("BISENET_RESULT (create_complete_style):")
        print(f"Keys: {list(bisenet_result.keys())}")
        if 'mask' in bisenet_result:
            mask = bisenet_result['mask']
            print(f"mask - shape: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}")
            unique_classes = np.unique(mask)
            print(f"mask - unique classes: {sorted(unique_classes.tolist())}")
            for class_id in range(19):
                count = np.sum(mask == class_id)
                if count > 0:
                    attr = bisenet_result.get('attribute_mapping', {}).get(class_id, 'unknown')
                    print(f"  class_id={class_id:2d} ({attr:10s}): {count:6d} pixels")
        if 'colorized_mask' in bisenet_result:
            cm = bisenet_result['colorized_mask']
            print(f"colorized_mask - shape: {cm.shape}, dtype: {cm.dtype}")
        if 'attributes' in bisenet_result:
            attrs = bisenet_result['attributes']
            print(f"attributes ({len(attrs)}): {attrs}")
        if 'attribute_mapping' in bisenet_result:
            am = bisenet_result['attribute_mapping']
            print(f"attribute_mapping - {len(am)} mappings (full): {am}")
        print("=" * 80)
        
        segmentation_mask = bisenet_result['mask']
        colorized_mask = bisenet_result['colorized_mask']
        attribute_mapping = bisenet_result['attribute_mapping']
        
        # Step 4: Style extraction
        image_rgb = np.array(cropped_face.convert('RGB'))
        style_data = extract_makeup_style(
            image_rgb=image_rgb,
            segmentation_mask=segmentation_mask,
            attribute_mapping=attribute_mapping
        )
        
        style_id = style_data.get('style_id')
        style_parameters = {
            "lips": style_data.get('lips', {}),
            "eyes": style_data.get('eyes', {}),
            "eyebrows": style_data.get('eyebrows', {}),
            "skin": style_data.get('skin', {}),
            "cheeks": style_data.get('cheeks', {}),
            "nose": style_data.get('nose', {})
        }
        
        # Create annotated image for thumbnail
        if colorized_mask.shape[:2] != image_rgb.shape[:2]:
            colorized_mask_resized = cv2.resize(
                colorized_mask, 
                (image_rgb.shape[1], image_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            colorized_mask_resized = colorized_mask
        annotated_image = cv2.addWeighted(image_rgb.copy(), 0.6, colorized_mask_resized, 0.4, 0)
        
        # Prepare style response data for later use
        colorized_pil = Image.fromarray(colorized_mask)
        colorized_buffer = BytesIO()
        colorized_pil.save(colorized_buffer, format="PNG")
        colorized_b64 = base64.b64encode(colorized_buffer.getvalue()).decode("utf-8")
        
        annotated_pil = Image.fromarray(annotated_image)
        annotated_buffer = BytesIO()
        annotated_pil.save(annotated_buffer, format="PNG")
        annotated_b64 = base64.b64encode(annotated_buffer.getvalue()).decode("utf-8")
        
        style_response = {
            'style_id': style_id,
            'style_parameters': style_parameters,
            'segmentation': {
                'colorized_mask': f"data:image/png;base64,{colorized_b64}",
                'annotated_image': f"data:image/png;base64,{annotated_b64}"
            },
            'face_detection': {
                'bounding_box': [x1, y1, x2, y2],
                'original_size': {'width': cropped_face.width, 'height': cropped_face.height}
            }
        }
    
        # Prepare complete style data for LUT/shader generation
        complete_style_data = {
            'style_id': style_id,
            **style_parameters
        }
    
        temp_dir = tempfile.mkdtemp(prefix=f"style_{style_id}_")
        masks_dir = os.path.join(temp_dir, "region_masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        # Step 5: Generate RGBA overlays for regions (placeholder, populated after masks build)
        overlay_files: Dict[str, str] = {}
        mask_urls: Dict[str, str] = {}
        minio_service = MinioService()
        
        # Step 6: Upload overlays and style parameters to storage (performed after overlays generated)
        style_params_path = os.path.join(temp_dir, f"{style_id}_params.json")
        params_object = f"styles/{style_id}/style_params.json"
        
        # Step 7: Generate filter preview on original image
        print(f"🎬 Step 7: Generating filter preview...")
        from app.services.filter_application import apply_style_to_image
        
        # Create face mask from segmentation (only apply filter to face region)
        # Combine all non-background regions as face mask
        face_mask = (segmentation_mask > 0).astype(np.float32)
        
        attribute_mapping = bisenet_result['attribute_mapping']

        def build_region_masks(seg_mask: np.ndarray, attr_map: Dict[int, str]) -> Dict[str, np.ndarray]:
            masks: Dict[str, np.ndarray] = {}
            if seg_mask is None or attr_map is None:
                return masks

            def _mask_from_class_ids(class_ids):
                out = np.zeros_like(seg_mask, dtype=np.float32)
                for cid in class_ids:
                    out[seg_mask == cid] = 1.0
                return out

            lips_upper_ids = [class_id for class_id, attr in attr_map.items() if attr in ['u_lip', 'upper_lip']]
            if lips_upper_ids:
                masks['lips_upper'] = _mask_from_class_ids(lips_upper_ids)

            lips_lower_ids = [class_id for class_id, attr in attr_map.items() if attr in ['l_lip', 'lower_lip']]
            if lips_lower_ids:
                masks['lips_lower'] = _mask_from_class_ids(lips_lower_ids)

            lips_full_ids = [class_id for class_id, attr in attr_map.items() if attr in ['u_lip', 'l_lip', 'mouth']]
            if lips_full_ids:
                masks['lips'] = _mask_from_class_ids(lips_full_ids)

            eyes_class_ids = [class_id for class_id, attr in attr_map.items() if attr in ['l_eye', 'r_eye']]
            if eyes_class_ids:
                masks['eyes'] = _mask_from_class_ids(eyes_class_ids)

            left_brow_ids = [class_id for class_id, attr in attr_map.items() if attr in ['l_brow', 'left_brow']]
            if left_brow_ids:
                masks['eyebrow_left'] = _mask_from_class_ids(left_brow_ids)

            right_brow_ids = [class_id for class_id, attr in attr_map.items() if attr in ['r_brow', 'right_brow']]
            if right_brow_ids:
                masks['eyebrow_right'] = _mask_from_class_ids(right_brow_ids)

            eyebrow_combined_ids = [class_id for class_id, attr in attr_map.items() if attr in ['l_brow', 'r_brow']]
            if eyebrow_combined_ids:
                masks['eyebrows'] = _mask_from_class_ids(eyebrow_combined_ids)
            elif 'eyebrow_left' in masks or 'eyebrow_right' in masks:
                left_mask = masks.get('eyebrow_left')
                right_mask = masks.get('eyebrow_right')
                if left_mask is not None and right_mask is not None:
                    masks['eyebrows'] = np.clip(left_mask + right_mask, 0.0, 1.0)
                else:
                    masks['eyebrows'] = left_mask if left_mask is not None else right_mask

            nose_ids = [class_id for class_id, attr in attr_map.items() if attr == 'nose']
            if nose_ids:
                masks['nose'] = _mask_from_class_ids(nose_ids)

            skin_class_ids = [class_id for class_id, attr in attr_map.items() if attr == 'skin']
            if skin_class_ids:
                masks['skin'] = _mask_from_class_ids(skin_class_ids)

            # Ensure all masks are float32 with values in [0, 1]
            for key, value in masks.items():
                masks[key] = np.clip(value.astype(np.float32), 0.0, 1.0)

            return masks

        region_masks_preview = build_region_masks(segmentation_mask, attribute_mapping)

        # Step 5: Generate RGBA overlays for regions
        print(f"🎨 Step 5: Generating RGBA overlays for {style_id}...")
        from app.services.rgba_mask_generation import (
            generate_region_overlays,
            compute_average_color,
            resize_mask_to_image,
        )

        region_meshes = compute_region_meshes(image_rgb)

        region_overlays = generate_region_overlays(
            image_rgb=image_rgb,
            region_masks=region_masks_preview,
            fill_with_original=True
        )

        for region_name, overlay_image in region_overlays.items():
            filename = f"{style_id}_{region_name}_mask.png"
            filepath = os.path.join(masks_dir, filename)
            overlay_image.save(filepath)
            overlay_files[region_name] = filepath

            avg_color = compute_average_color(
                image_rgb,
                resize_mask_to_image(region_masks_preview.get(region_name, np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)), image_rgb.shape)
            )
            if avg_color is not None:
                if region_name in complete_style_data:
                    region_entry = complete_style_data[region_name]
                    if isinstance(region_entry, dict):
                        region_entry['average_rgb'] = avg_color
                    else:
                        complete_style_data[region_name] = {'average_rgb': avg_color}
                else:
                    complete_style_data[region_name] = {'average_rgb': avg_color}

        # Step 6: Upload overlays and style parameters to storage
        print(f"📦 Step 6: Uploading RGBA overlays to storage...")
        for region_name, overlay_path in overlay_files.items():
            object_name = f"styles/{style_id}/region_masks/{os.path.basename(overlay_path)}"
            await minio_service.upload_file(overlay_path, object_name)
            mask_urls[region_name] = minio_service.get_file_url(object_name, expires=86400*7, use_proxy=True)

        with open(style_params_path, 'w') as f:
            json.dump(complete_style_data, f, indent=2)

        await minio_service.upload_file(style_params_path, params_object)
        params_url = minio_service.get_file_url(params_object, expires=86400*7, use_proxy=True)
        
        # Determine preview image for filter application (default non_makeup face)
        DEFAULT_PREVIEW_IMAGE_PATH = "../docs/non_makeup.jpg"
        preview_image_rgb: Optional[np.ndarray] = None
        preview_face_mask: Optional[np.ndarray] = None
        preview_region_masks: Dict[str, np.ndarray] = {}

        if preview_temp_path and os.path.exists(preview_temp_path):
            try:
                with Image.open(preview_temp_path) as preview_img:
                    preview_image_rgb = np.array(preview_img.convert('RGB'))

                preview_bisenet = process_image_bisenet(
                    preview_temp_path,
                    device=device,
                    return_mask_array=True
                )

                if preview_bisenet and 'mask' in preview_bisenet:
                    preview_seg_mask = preview_bisenet['mask']
                    if preview_seg_mask.shape[:2] != preview_image_rgb.shape[:2]:
                        preview_seg_mask = cv2.resize(
                            preview_seg_mask.astype(np.uint8),
                            (preview_image_rgb.shape[1], preview_image_rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(preview_seg_mask.dtype)
                    preview_face_mask = (preview_seg_mask > 0).astype(np.float32)
                    preview_region_masks = build_region_masks(
                        preview_seg_mask,
                        preview_bisenet.get('attribute_mapping', {})
                    )
            except Exception as preview_error:
                print(f"Warning: Failed to generate user preview: {preview_error}")
                preview_image_rgb = None
                preview_face_mask = None
                preview_region_masks = {}

        if preview_image_rgb is None and os.path.exists(DEFAULT_PREVIEW_IMAGE_PATH):
            try:
                with Image.open(DEFAULT_PREVIEW_IMAGE_PATH) as preview_img:
                    preview_image_rgb = np.array(preview_img.convert('RGB'))

                preview_bisenet = process_image_bisenet(
                    DEFAULT_PREVIEW_IMAGE_PATH,
                    device=device,
                    return_mask_array=True
                )

                if preview_bisenet and 'mask' in preview_bisenet:
                    preview_seg_mask = preview_bisenet['mask']
                    if preview_seg_mask.shape[:2] != preview_image_rgb.shape[:2]:
                        preview_seg_mask = cv2.resize(
                            preview_seg_mask.astype(np.uint8),
                            (preview_image_rgb.shape[1], preview_image_rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(preview_seg_mask.dtype)
                    preview_face_mask = (preview_seg_mask > 0).astype(np.float32)
                    preview_region_masks = build_region_masks(
                        preview_seg_mask,
                        preview_bisenet.get('attribute_mapping', {})
                    )
            except Exception as preview_error:
                print(f"Warning: Failed to generate default preview: {preview_error}")

        if preview_image_rgb is None:
            # Fallback to original cropped face if default preview fails
            preview_image_rgb = image_rgb
            preview_face_mask = face_mask
            preview_region_masks = region_masks_preview

        if preview_face_mask is None:
            preview_face_mask = np.ones(preview_image_rgb.shape[:2], dtype=np.float32)

        preview_style_payload = {
            'download_urls': {
                'region_masks': mask_urls
            },
            **complete_style_data,
            'metadata': {
                'region_meshes': region_meshes
            }
        }

        # Apply filter to preview image with generated masks
        filtered_preview = apply_style_to_image(
            preview_image_rgb,
            preview_style_payload,
            intensity=1.0,
            face_mask=preview_face_mask,
            use_regions=len(preview_region_masks) > 0,
            region_masks=preview_region_masks if preview_region_masks else None
        )
        
        # Convert filtered preview to base64
        filtered_preview_pil = Image.fromarray(filtered_preview)
        filtered_buffer = BytesIO()
        filtered_preview_pil.save(filtered_buffer, format="PNG")
        filtered_preview_b64 = base64.b64encode(filtered_buffer.getvalue()).decode("utf-8")
        filtered_preview_data_url = f"data:image/png;base64,{filtered_preview_b64}"
        
        # Convert cropped image (image_rgb) to base64 for Filter Preview display
        cropped_image_pil = Image.fromarray(image_rgb)
        cropped_buffer = BytesIO()
        cropped_image_pil.save(cropped_buffer, format="PNG")
        cropped_image_b64 = base64.b64encode(cropped_buffer.getvalue()).decode("utf-8")
        cropped_image_data_url = f"data:image/png;base64,{cropped_image_b64}"
        
        # Generate mask preview images for display (on white background, no overlay)
        mask_previews = {}
        
        # First, add BiSeNet raw results (segmentation mask and colorized mask)
        if bisenet_result:
            print(f"\n🔍 ========== BISENET RAW RESULTS PREVIEW ==========")
            
            # Get original BiSeNet outputs
            raw_segmentation_mask = bisenet_result['mask']  # 512x512, uint8, 0-18
            raw_colorized_mask = bisenet_result['colorized_mask']  # 512x512x3, uint8, RGB
            
            # Render segmentation mask from class IDs using PALETTE when available
            try:
                from ai_models.BiseNet.inference_bisenet import PALETTE
            except ImportError:
                PALETTE = None
            if PALETTE and len(PALETTE) >= 19:
                seg_mask_vis_rgb = np.zeros((*raw_segmentation_mask.shape, 3), dtype=np.uint8)
                for class_id in np.unique(raw_segmentation_mask):
                    mask = (raw_segmentation_mask == class_id)
                    if np.any(mask) and class_id < len(PALETTE):
                        seg_mask_vis_rgb[mask] = PALETTE[class_id]
            else:
                # Fallback: simple grayscale by class id
                seg_mask_vis = (raw_segmentation_mask.astype(np.float32) / 18.0 * 255).astype(np.uint8)
                seg_mask_vis_rgb = np.stack([seg_mask_vis, seg_mask_vis, seg_mask_vis], axis=-1)
            seg_mask_pil = Image.fromarray(seg_mask_vis_rgb)
            seg_mask_buffer = BytesIO()
            seg_mask_pil.save(seg_mask_buffer, format="PNG")
            seg_mask_b64 = base64.b64encode(seg_mask_buffer.getvalue()).decode("utf-8")
            mask_previews['bisenet_segmentation_mask'] = f"data:image/png;base64,{seg_mask_b64}"
            print(f"   ✅ BiSeNet segmentation mask preview generated (from class IDs)")
            
            # Raw colorized mask from BiSeNet
            colorized_mask_pil = Image.fromarray(raw_colorized_mask)
            colorized_mask_buffer = BytesIO()
            colorized_mask_pil.save(colorized_mask_buffer, format="PNG")
            colorized_mask_b64 = base64.b64encode(colorized_mask_buffer.getvalue()).decode("utf-8")
            mask_previews['bisenet_colorized_mask'] = f"data:image/png;base64,{colorized_mask_b64}"
            print(f"   ✅ BiSeNet colorized mask preview generated")
            print(f"🔍 =====================================================\n")
        
        # Generate individual region mask previews (on white background)
        if region_masks_preview:
            print(f"\n🔍 ========== MASK PREVIEW GENERATION (CREATE STYLE) ==========")
            print(f"🔍 Generating mask previews for {len(region_masks_preview)} regions: {list(region_masks_preview.keys())}")
            
            # Use a standard display size (512x512 for consistency)
            display_h, display_w = 512, 512
            mask_h, mask_w = segmentation_mask.shape[:2]
            img_h, img_w = image_rgb.shape[:2]
            
            # Create white background
            white_bg = np.ones((display_h, display_w, 3), dtype=np.uint8) * 255
            
            # Create visualization for each region mask (use original cropped image colors, not filtered)
            # Use image_rgb (original cropped image) instead of filtered_preview
            original_for_display = image_rgb
            if original_for_display.shape[:2] != (display_h, display_w):
                original_for_display = cv2.resize(
                    original_for_display,
                    (display_w, display_h),
                    interpolation=cv2.INTER_LINEAR
                )
            
            for region_name, region_mask in region_masks_preview.items():
                # Resize mask to display size
                if region_mask.shape[:2] != (display_h, display_w):
                    region_mask_resized = cv2.resize(
                        region_mask.astype(np.float32),
                        (display_w, display_h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.float32)
                else:
                    region_mask_resized = region_mask
                
                # Build region preview using ORIGINAL cropped image colors on white background
                mask_bool = (region_mask_resized > 0.5)
                mask_bool_3 = np.repeat(mask_bool[:, :, None], 3, axis=2)
                mask_colored = white_bg.copy()
                mask_colored[mask_bool_3] = original_for_display[mask_bool_3]
                
                # Convert to base64
                mask_pil = Image.fromarray(mask_colored)
                mask_buffer = BytesIO()
                mask_pil.save(mask_buffer, format="PNG")
                mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")
                mask_previews[region_name] = f"data:image/png;base64,{mask_b64}"
                print(f"   ✅ {region_name} preview generated successfully")
            
            print(f"🔍 ==========================================================\n")
        
        # Step 6b: Store style metadata (file-based for now)
        response_data = {
            "success": True,
            "style_id": style_id,
            "name": name or f"Style {style_id}",
            "description": description,
            "download_urls": {
                "region_masks": mask_urls,
                "style_parameters": params_url
            },
            "style_parameters": complete_style_data,
            "storage_info": {
                "minio_bucket": settings.MINIO_BUCKET,
                "storage_path": f"styles/{style_id}/",
                "database_stored": store_in_db
            },
            "metadata": {
                "filter_preview": filtered_preview_data_url,
                "original_cropped": cropped_image_data_url,
                "segmentation": style_response['segmentation'],
                "face_detection": style_response['face_detection'],
                "region_meshes": region_meshes
            },
            "mask_previews": mask_previews,  # Individual region masks for display
            "regions_detected": list(region_masks_preview.keys()) if region_masks_preview else [],
            "created_at": datetime.now().isoformat()
        }
        
        # Save to file-based storage
        save_style(style_id, response_data)
        print(f"💾 Style saved: {style_id}")
        
        return response_data
        
    except HTTPException:
        # Re-raise HTTPExceptions (like "No faces detected")
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error in create_complete_style: {error_trace}")
        error_msg = str(e) if str(e) else type(e).__name__
        raise HTTPException(
            status_code=500,
            detail=f"Style creation failed: {error_msg}"
        )
    finally:
        # Cleanup temporary files
        for path in [temp_path, temp_path.replace(".jpg", "_cropped.jpg")]:
            if os.path.exists(path):
                os.remove(path)
        if preview_temp_path and os.path.exists(preview_temp_path):
            os.remove(preview_temp_path)
        # Cleanup temporary directory if it exists
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.post("/style/apply_filter")
async def apply_filter_to_image(
    file: UploadFile = File(...),
    style_id: str = Form(...),
    intensity: float = Form(1.0)
):
    """
    Apply makeup filter to uploaded image
    
    Args:
        file: Image file to apply filter to
        style_id: Style ID to use
        intensity: Filter intensity (0.0 to 1.0)
    
    Returns:
        Base64 encoded filtered image
    """
    from app.services.style_storage import get_style
    from app.services.filter_application import apply_style_to_image_file
    import base64
    
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    # Get style data
    style_data = get_style(style_id)
    if not style_data:
        raise HTTPException(status_code=404, detail=f"Style {style_id} not found")
    
    # Validate intensity
    intensity = max(0.0, min(1.0, intensity))
    
    # Save uploaded file temporarily
    temp_filename = f"filter_apply_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join("/tmp", temp_filename)
    
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Detect face and run BiSeNet segmentation for region masks
        from retinaface import RetinaFace
        from ai_models.BiseNet.inference_bisenet import process_image_bisenet
        import cv2
        import numpy as np
        import torch
        from app.services.filter_application import apply_style_to_image
        
        # Initialize variables for BiSeNet previews
        bisenet_result = None
        colorized_mask = None
        attribute_mapping = {}
        
        # Get full image dimensions first
        with Image.open(temp_path) as img:
            h, w = img.size[1], img.size[0]
        
        face_results = RetinaFace.detect_faces(temp_path)
        if not face_results:
            # No face detected, apply filter to entire image
            face_mask = np.ones((h, w), dtype=np.float32)
            region_masks = None
            bisenet_result = None
        else:
            # Get first face for bounding box (but don't crop yet)
            first_face = list(face_results.values())[0]
            face_box = first_face["facial_area"]
            x1, y1, x2, y2 = [int(coord) for coord in face_box]
            
            # Run BiSeNet segmentation on FULL IMAGE (not cropped)
            # Use same device logic as segmentation page for consistency
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            bisenet_result = process_image_bisenet(temp_path, device=device, return_mask_array=True)
            
            if bisenet_result:
                # ====================================================================
                # LOG CHI TIẾT bisenet_result NGAY SAU KHI NHẬN ĐƯỢC
                # ====================================================================
                print(f"\n🔍 ========== BISENET_RESULT CHI TIẾT ==========")
                print(f"🔍 Keys trong bisenet_result: {list(bisenet_result.keys())}")
                if 'attributes' in bisenet_result:
                    attrs = bisenet_result['attributes']
                    print(f"🔍 attributes ({len(attrs)}): {attrs}")
                if 'attribute_mapping' in bisenet_result:
                    am = bisenet_result['attribute_mapping']
                    print(f"🔍 attribute_mapping ({len(am)} mappings): {am}")
                
                # Log segmentation_mask
                if 'mask' in bisenet_result:
                    seg_mask_raw = bisenet_result['mask']
                    print(f"\n🔍 1. bisenet_result['mask'] (segmentation_mask):")
                    print(f"   - Type: {type(seg_mask_raw)}")
                    print(f"   - Shape: {seg_mask_raw.shape if hasattr(seg_mask_raw, 'shape') else 'N/A'}")
                    print(f"   - Dtype: {seg_mask_raw.dtype if hasattr(seg_mask_raw, 'dtype') else 'N/A'}")
                    if hasattr(seg_mask_raw, 'min') and hasattr(seg_mask_raw, 'max'):
                        print(f"   - Min: {seg_mask_raw.min()}, Max: {seg_mask_raw.max()}")
                    if hasattr(seg_mask_raw, 'size'):
                        print(f"   - Total pixels: {seg_mask_raw.size}")
                    if hasattr(seg_mask_raw, '__array__'):
                        unique_vals = np.unique(seg_mask_raw)
                        print(f"   - Unique values: {sorted(unique_vals)}")
                        print(f"   - Number of unique classes: {len(unique_vals)}")
                        # Sample một số giá trị từ góc trên bên trái
                        sample_size = min(10, seg_mask_raw.shape[0])
                        sample_vals = seg_mask_raw[:sample_size, :sample_size].flatten()[:20]
                        print(f"   - Sample values (top-left 10x10, first 20): {sample_vals.tolist()}")
                else:
                    print(f"   ❌ 'mask' NOT in bisenet_result!")
                
                # Log colorized_mask
                if 'colorized_mask' in bisenet_result:
                    color_mask_raw = bisenet_result['colorized_mask']
                    print(f"\n🔍 2. bisenet_result['colorized_mask']:")
                    print(f"   - Type: {type(color_mask_raw)}")
                    print(f"   - Shape: {color_mask_raw.shape if hasattr(color_mask_raw, 'shape') else 'N/A'}")
                    print(f"   - Dtype: {color_mask_raw.dtype if hasattr(color_mask_raw, 'dtype') else 'N/A'}")
                    if hasattr(color_mask_raw, 'min') and hasattr(color_mask_raw, 'max'):
                        print(f"   - Min: {color_mask_raw.min()}, Max: {color_mask_raw.max()}")
                    if hasattr(color_mask_raw, 'size'):
                        print(f"   - Total pixels: {color_mask_raw.size}")
                    if hasattr(color_mask_raw, '__array__') and len(color_mask_raw.shape) >= 2:
                        # Sample một số giá trị RGB từ góc trên bên trái
                        sample_size = min(5, color_mask_raw.shape[0])
                        sample_rgb = color_mask_raw[:sample_size, :sample_size].reshape(-1, 3)[:10]
                        print(f"   - Sample RGB values (top-left 5x5, first 10):")
                        for i, rgb in enumerate(sample_rgb):
                            print(f"      [{i}]: RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")
                else:
                    print(f"   ❌ 'colorized_mask' NOT in bisenet_result!")
                
                # Log attribute_mapping - CHI TIẾT
                if 'attribute_mapping' in bisenet_result:
                    attr_mapping = bisenet_result['attribute_mapping']
                    print(f"\n🔍 3. bisenet_result['attribute_mapping']:")
                    print(f"   - Type: {type(attr_mapping)}")
                    print(f"   - Keys (class_ids): {sorted(attr_mapping.keys()) if isinstance(attr_mapping, dict) else 'N/A'}")
                    if isinstance(attr_mapping, dict):
                        print(f"   - Total mappings: {len(attr_mapping)}")
                        print(f"   - Mapping (class_id → attribute_name):")
                        print(f"      {'Class ID':<10} {'Attribute Name':<15} {'Pixel Count':<12}")
                        print(f"      {'-'*40}")
                        for class_id in sorted(attr_mapping.keys()):
                            attr_name = attr_mapping[class_id]
                            # Count pixels for this class_id if segmentation_mask is available
                            pixel_count = "N/A"
                            if 'mask' in bisenet_result:
                                seg_mask = bisenet_result['mask']
                                if hasattr(seg_mask, '__array__'):
                                    pixel_count = np.sum(seg_mask == class_id)
                            print(f"      {class_id:<10} {attr_name:<15} {pixel_count:<12}")
                        
                        # Also print reverse mapping (attribute_name → class_id) for important attributes
                        print(f"\n   - Reverse mapping (attribute_name → class_id) for important attributes:")
                        important_attrs = ['nose', 'l_brow', 'r_brow', 'u_lip', 'l_lip', 'l_eye', 'r_eye', 'skin']
                        for attr in important_attrs:
                            class_ids_for_attr = [cid for cid, attr_name in attr_mapping.items() if attr_name == attr]
                            if class_ids_for_attr:
                                print(f"      {attr:10s} → class_ids: {class_ids_for_attr}")
                            else:
                                print(f"      {attr:10s} → NOT FOUND in attribute_mapping")
                        
                        # Check for expected mappings (based on CELEBA_ATTRIBUTES)
                        print(f"\n   - Expected mappings verification:")
                        expected_mappings = {
                            10: 'nose',
                            2: 'l_brow',
                            3: 'r_brow',
                            12: 'u_lip',
                            13: 'l_lip',
                            4: 'l_eye',
                            5: 'r_eye',
                            1: 'skin'
                        }
                        for class_id, expected_attr in expected_mappings.items():
                            actual_attr = attr_mapping.get(class_id, 'NOT_IN_MAPPING')
                            status = "✅" if actual_attr == expected_attr else "❌"
                            print(f"      {status} class_id={class_id:2d}: expected '{expected_attr:10s}', got '{actual_attr:10s}'")
                            if actual_attr != expected_attr and actual_attr != 'NOT_IN_MAPPING':
                                print(f"         ⚠️  WARNING: Mapping mismatch!")
                else:
                    print(f"   ❌ 'attribute_mapping' NOT in bisenet_result!")
                
                print(f"🔍 ===========================================\n")
                
                # Gán vào biến
                segmentation_mask = bisenet_result['mask']  # Raw segmentation mask từ BiSeNet
                attribute_mapping = bisenet_result['attribute_mapping']  # class_id → attribute_name
                colorized_mask = bisenet_result['colorized_mask']  # Raw colorized mask từ BiSeNet (đã được colorize sẵn)
                
                # ====================================================================
                # LOG RAW BISENET RESULTS (BEFORE ANY PROCESSING)
                # ====================================================================
                print(f"\n📊 ========== RAW BISENET RESULTS (TRƯỚC XỬ LÝ) ==========")
                print(f"📊 1. Segmentation Mask (Raw từ BiSeNet):")
                print(f"   - Shape: {segmentation_mask.shape}")
                print(f"   - Dtype: {segmentation_mask.dtype}")
                print(f"   - Min: {segmentation_mask.min()}, Max: {segmentation_mask.max()}")
                print(f"   - Unique class IDs: {sorted(np.unique(segmentation_mask))}")
                print(f"   - Total pixels: {segmentation_mask.size}")
                
                # ====================================================================
                # LOG TẤT CẢ CLASS IDs VÀ PIXEL COUNTS (0-18)
                # ====================================================================
                print(f"\n📊 Chi tiết pixel counts cho TẤT CẢ class IDs (0-18):")
                print(f"   {'Class ID':<10} {'Attribute Name':<15} {'Pixel Count':<12} {'Status':<6} {'RGB Color':<20}")
                print(f"   {'-'*70}")
                
                # Import PALETTE để hiển thị màu
                try:
                    from ai_models.BiseNet.inference_bisenet import PALETTE, CELEBA_ATTRIBUTES
                except ImportError:
                    PALETTE = None
                    CELEBA_ATTRIBUTES = []
                
                # Log tất cả class IDs từ 0-18 (19 classes: 18 attributes + background)
                for class_id in range(19):  # 0-18
                    pixel_count = np.sum(segmentation_mask == class_id)
                    status = "✅" if pixel_count > 0 else "❌"
                    
                    # Get attribute name
                    if class_id == 0:
                        attr_name = "background"
                    elif class_id <= len(CELEBA_ATTRIBUTES):
                        attr_name = CELEBA_ATTRIBUTES[class_id - 1]
                    else:
                        attr_name = f"class_{class_id}"
                    
                    # Get RGB color from PALETTE
                    if PALETTE and class_id < len(PALETTE):
                        rgb_color = PALETTE[class_id]
                        rgb_str = f"RGB({rgb_color[0]},{rgb_color[1]},{rgb_color[2]})"
                    else:
                        rgb_str = "N/A"
                    
                    print(f"   {class_id:<10} {attr_name:<15} {pixel_count:<12} {status:<6} {rgb_str:<20}")
                
                # Count pixels for each important class_id BEFORE any processing
                important_class_ids = {2: 'nose', 6: 'l_brow', 7: 'r_brow', 11: 'u_lip', 12: 'l_lip'}
                print(f"\n📊 Tóm tắt pixel counts cho classes quan trọng (BEFORE processing):")
                for class_id, attr_name in sorted(important_class_ids.items()):
                    pixel_count = np.sum(segmentation_mask == class_id)
                    status = "✅" if pixel_count > 0 else "❌"
                    print(f"   {status} class_id={class_id:2d} ({attr_name:8s}): {pixel_count:6d} pixels")
                
                print(f"\n📊 2. Colorized Mask (Raw từ BiSeNet):")
                print(f"   - Shape: {colorized_mask.shape}")
                print(f"   - Dtype: {colorized_mask.dtype}")
                print(f"   - Min: {colorized_mask.min()}, Max: {colorized_mask.max()}")
                
                print(f"📊 =======================================================\n")
                
                # CRITICAL: Ensure mask is uint8 and properly formatted
                # Check dtype before any processing
                print(f"🔍 Processing mask: dtype check and conversion...")
                
                if segmentation_mask.dtype != np.uint8:
                    print(f"⚠️  Converting mask from {segmentation_mask.dtype} to uint8")
                    # Preserve values when converting
                    if segmentation_mask.dtype in [np.int32, np.int64]:
                        # Ensure values are in valid range
                        segmentation_mask = np.clip(segmentation_mask, 0, 255).astype(np.uint8)
                    else:
                        segmentation_mask = segmentation_mask.astype(np.uint8)
                
                # Debug: Check original segmentation mask BEFORE any resize
                original_shape = segmentation_mask.shape[:2]
                original_classes = np.unique(segmentation_mask)
                print(f"🔍 Original segmentation mask (after dtype fix): shape={original_shape}, dtype={segmentation_mask.dtype}")
                print(f"🔍 Original detected classes: {sorted(original_classes)}")
                print(f"🔍 Original image size (h, w): ({h}, {w})")
                
                # Check for required classes BEFORE resize - with pixel counts
                required_classes = {2, 3, 10, 12, 13}  # nose, l_brow, r_brow, u_lip, l_lip
                found_before_resize = required_classes.intersection(set(original_classes))
                print(f"🔍 Required classes found BEFORE resize: {sorted(found_before_resize)}")
                missing_before_resize = required_classes - found_before_resize
                if missing_before_resize:
                    print(f"⚠️  Missing classes BEFORE resize: {sorted(missing_before_resize)}")
                    # Check pixel counts for missing classes in ORIGINAL mask (may have very few pixels)
                    for missing_id in sorted(missing_before_resize):
                        pixel_count = np.sum(segmentation_mask == missing_id)
                        attr_name = attribute_mapping.get(missing_id, f'class_{missing_id}')
                        print(f"      class_id={missing_id} ({attr_name}): {pixel_count} pixels in original 512x512 mask")
                        if pixel_count > 0:
                            print(f"         ⚠️  WARNING: class_id={missing_id} HAS pixels but not in np.unique() - dtype issue?")
                else:
                    print(f"✅ All required classes found in original mask!")
                
                # OPTIMIZATION: Keep mask at original BiSeNet size (512x512) to preserve small regions
                # Only resize when absolutely necessary (for preview/display)
                # When applying filter, we'll resize image to match mask size, then resize result back
                
                # Store original mask size for later use
                mask_h, mask_w = segmentation_mask.shape[:2]
                print(f"✅ Keeping segmentation mask at BiSeNet output size: ({mask_h}, {mask_w})")
                print(f"   Original image size: ({h}, {w})")
                print(f"   Mask will be kept at high resolution to preserve small regions (eyebrows, lips)")
                
                # Check pixel counts for required classes in ORIGINAL mask (no resize yet)
                print(f"🔍 Pixel counts for required classes in ORIGINAL 512x512 mask:")
                for req_id in sorted(required_classes):
                    pixel_count = np.sum(segmentation_mask == req_id)
                    attr_name = attribute_mapping.get(req_id, f'class_{req_id}')
                    status = "✅" if pixel_count > 0 else "❌"
                    print(f"   {status} class_id={req_id:2d} ({attr_name:8s}): {pixel_count:6d} pixels")
                
                # ====================================================================
                # LOG BEFORE RESIZE
                # ====================================================================
                print(f"\n🔄 ========== BEFORE RESIZE ==========")
                print(f"🔄 Original image size: ({h}, {w})")
                print(f"🔄 Segmentation mask size (sẽ giữ nguyên): {segmentation_mask.shape}")
                print(f"🔄 Colorized mask size (sẽ resize): {colorized_mask.shape}")
                print(f"🔄 =====================================\n")
                
                # Only resize colorized_mask for preview (but keep segmentation_mask at 512x512)
                # Colorized mask is only for visualization, so resize is OK
                if colorized_mask.shape[:2] != (h, w):
                    print(f"🔄 Resizing colorized_mask from {colorized_mask.shape[:2]} to ({h}, {w})...")
                    colorized_mask_resized_for_preview = cv2.resize(
                        colorized_mask,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    print(f"✅ Colorized mask resized: {colorized_mask_resized_for_preview.shape}")
                else:
                    print(f"ℹ️  Colorized mask đã có kích thước đúng, không cần resize")
                    colorized_mask_resized_for_preview = colorized_mask
                
                # Store original colorized_mask for later use at full resolution
                colorized_mask_original = colorized_mask.copy()
                
                # ====================================================================
                # LOG AFTER RESIZE
                # ====================================================================
                print(f"\n✅ ========== AFTER RESIZE ==========")
                print(f"✅ Segmentation mask (giữ nguyên): {segmentation_mask.shape}")
                print(f"✅ Colorized mask original (backup): {colorized_mask_original.shape}")
                print(f"✅ Colorized mask resized (for preview): {colorized_mask_resized_for_preview.shape}")
                print(f"✅ ====================================\n")
                
                # ====================================================================
                # RAW BISENET RESULTS PREVIEW (before any processing)
                # ====================================================================
                # Hiển thị kết quả TRỰC TIẾP từ BiSeNet trước khi tạo masks và annotated image
                print(f"\n📊 ========== RAW BISENET RESULTS ==========")
                print(f"📊 Segmentation mask (raw):")
                print(f"   Shape: {segmentation_mask.shape}")
                print(f"   Dtype: {segmentation_mask.dtype}")
                print(f"   Min/Max: {segmentation_mask.min()}/{segmentation_mask.max()}")
                print(f"   Unique class IDs: {sorted(np.unique(segmentation_mask))}")
                print(f"   Total pixels: {segmentation_mask.size}")
                
                print(f"\n📊 Colorized mask (raw):")
                print(f"   Shape: {colorized_mask.shape}")
                print(f"   Dtype: {colorized_mask.dtype}")
                print(f"   Min/Max: {colorized_mask.min()}/{colorized_mask.max()}")
                
                # Create raw BiSeNet previews (segmentation mask visualization and colorized mask)
                raw_bisenet_previews = {}
                
                # ====================================================================
                # 1. Visualize raw segmentation_mask với PALETTE colors (Class IDs)
                # ====================================================================
                print(f"\n🎨 Tạo Raw Segmentation Mask visualization với PALETTE colors...")
                
                # Import PALETTE nếu chưa import
                if 'PALETTE' not in locals() or PALETTE is None:
                    try:
                        from ai_models.BiseNet.inference_bisenet import PALETTE
                    except ImportError:
                        # Fallback: tạo grayscale nếu không có PALETTE
                        print(f"   ⚠️  PALETTE không tìm thấy, dùng grayscale")
                        seg_mask_vis = (segmentation_mask.astype(np.float32) / 19.0 * 255).astype(np.uint8)
                        seg_mask_vis_rgb = np.stack([seg_mask_vis, seg_mask_vis, seg_mask_vis], axis=-1)
                        PALETTE = None
                
                if PALETTE and len(PALETTE) >= 20:
                    # Dùng PALETTE để colorize theo class IDs
                    seg_mask_vis_rgb = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)
                    for class_id in range(len(PALETTE)):
                        mask = (segmentation_mask == class_id)
                        if np.any(mask):
                            seg_mask_vis_rgb[mask] = PALETTE[class_id]
                            pixel_count = np.sum(mask)
                            attr_name = attribute_mapping.get(class_id, f'class_{class_id}')
                            print(f"   ✅ class_id={class_id:2d} ({attr_name:8s}): {pixel_count:6d} pixels → Color: RGB{PALETTE[class_id]}")
                    
                    print(f"   ✅ Raw Segmentation Mask visualization created với PALETTE colors")
                else:
                    # Fallback: grayscale
                    print(f"   ⚠️  Dùng grayscale visualization (PALETTE không available)")
                    seg_mask_vis = (segmentation_mask.astype(np.float32) / 19.0 * 255).astype(np.uint8)
                    seg_mask_vis_rgb = np.stack([seg_mask_vis, seg_mask_vis, seg_mask_vis], axis=-1)
                
                # Log thông số để vẽ Raw Segmentation Mask
                print(f"\n📐 Thông số để vẽ Raw Segmentation Mask (Class IDs):")
                print(f"   - Kích thước: {seg_mask_vis_rgb.shape}")
                print(f"   - Dtype: {seg_mask_vis_rgb.dtype}")
                print(f"   - Min/Max RGB: {seg_mask_vis_rgb.min()}/{seg_mask_vis_rgb.max()}")
                print(f"   - Unique class IDs trong mask: {sorted(np.unique(segmentation_mask))}")
                
                # Log color mapping cho từng class ID
                if PALETTE:
                    print(f"   - Color mapping (class_id → RGB):")
                    for class_id in sorted(np.unique(segmentation_mask)):
                        if class_id < len(PALETTE):
                            rgb = PALETTE[class_id]
                            attr_name = attribute_mapping.get(class_id, f'class_{class_id}')
                            pixel_count = np.sum(segmentation_mask == class_id)
                            print(f"      class_id={class_id:2d} ({attr_name:10s}) → RGB({rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d}) - {pixel_count:6d} pixels")
                
                # Save to base64
                seg_mask_pil = Image.fromarray(seg_mask_vis_rgb)
                seg_mask_buffer = BytesIO()
                seg_mask_pil.save(seg_mask_buffer, format="PNG")
                seg_mask_b64 = base64.b64encode(seg_mask_buffer.getvalue()).decode("utf-8")
                raw_bisenet_previews['raw_segmentation_mask'] = f"data:image/png;base64,{seg_mask_b64}"
                print(f"   ✅ Saved Raw Segmentation Mask visualization ({len(seg_mask_b64)} bytes)")
                
                # 2. Raw colorized_mask (before any resize or blend)
                colorized_pil = Image.fromarray(colorized_mask)
                colorized_buffer = BytesIO()
                colorized_pil.save(colorized_buffer, format="PNG")
                colorized_b64 = base64.b64encode(colorized_buffer.getvalue()).decode("utf-8")
                raw_bisenet_previews['raw_colorized_mask'] = f"data:image/png;base64,{colorized_b64}"
                
                print(f"   ✅ Raw BiSeNet previews created: {list(raw_bisenet_previews.keys())}")
                print(f"📊 ===========================================\n")
                
                # Map CelebAMask-HQ attributes to makeup regions
                # lips: u_lip, l_lip, mouth
                # eyes: l_eye, r_eye, eye_g
                # eyebrows: l_brow, r_brow
                # skin: skin
                # cheeks: (may need interpolation)
                
                from app.services.style_extraction import REGION_MAPPING
                
                # ====================================================================
                # TẠO REGION MASKS TỪ SEGMENTATION MASK GỐC (TRƯỚC KHI RESIZE)
                # ====================================================================
                print(f"\n📝 ========== TẠO REGION MASKS TỪ SEGMENTATION MASK GỐC ==========")
                print(f"📝 QUAN TRỌNG: Tạo masks từ segmentation_mask GỐC ({segmentation_mask.shape})")
                print(f"📝 Segmentation mask sẽ GIỮ NGUYÊN ở kích thước {segmentation_mask.shape}")
                print(f"📝 Sau khi tạo masks, sẽ resize masks về kích thước ảnh gốc ({h}, {w})")
                print(f"📝 Điều này đảm bảo masks được tạo chính xác từ kết quả BiSeNet gốc")
                print(f"📝 ================================================================\n")
                
                # Create region masks from segmentation at ORIGINAL mask size (512x512)
                # This preserves small regions like eyebrows and lips
                # CRITICAL: Tạo masks từ segmentation_mask GỐC trước, sau đó mới resize
                region_masks = {}
                region_masks_original_size = {}  # Store masks at original BiSeNet size (512x512)
                
                # Debug: Print detected attributes for debugging
                detected_classes = np.unique(segmentation_mask)
                print(f"🔍 Segmentation mask shape (kept at original): {segmentation_mask.shape}, detected class IDs (np.unique): {detected_classes}")
                
                # CRITICAL: Also check pixel counts directly (not just np.unique())
                # np.unique() might miss classes with very few pixels
                detected_by_pixels = set()
                for class_id in range(19):  # 0-18 for all possible CelebAMask-HQ classes (18 attributes + background)
                    pixel_count = np.sum(segmentation_mask == class_id)
                    if pixel_count > 0:
                        detected_by_pixels.add(class_id)
                        if class_id not in detected_classes:
                            attr_name = attribute_mapping.get(class_id, f'class_{class_id}')
                            print(f"   ⚠️  class_id={class_id} ({attr_name}) has {pixel_count} pixels but NOT in np.unique()!")
                
                print(f"🔍 Detected class IDs (by pixel count): {sorted(detected_by_pixels)}")
                print(f"🔍 Attribute mapping: {attribute_mapping}")
                print(f"🔍 Total pixels in segmentation mask: {segmentation_mask.size}")
                
                # Debug: Check specific attributes BEFORE creating masks
                print(f"\n🔍 ========== DETAILED ATTRIBUTE CHECK ==========")
                print(f"🔍 Checking for lip attributes:")
                for class_id in [12, 13]:  # u_lip=11, l_lip=12 based on CELEBA_ATTRIBUTES
                    in_mapping = class_id in attribute_mapping
                    in_detected = class_id in detected_classes
                    # CRITICAL: Always check pixel count directly, not just if in detected_classes
                    pixel_count = np.sum(segmentation_mask == class_id)
                    attr_name = attribute_mapping.get(class_id, 'NOT_IN_MAPPING')
                    status = "✅" if pixel_count > 0 else "❌"
                    print(f"   {status} class_id={class_id}:")
                    print(f"      - In attribute_mapping: {in_mapping} ({attr_name})")
                    print(f"      - In detected_classes (np.unique): {in_detected}")
                    print(f"      - Pixel count (direct check): {pixel_count}")
                    if pixel_count > 0 and not in_detected:
                        print(f"      ⚠️  WARNING: Has {pixel_count} pixels but NOT in np.unique()!")
                
                print(f"🔍 Checking for eyebrow attributes:")
                for class_id in [2, 3]:  # l_brow=6, r_brow=7
                    in_mapping = class_id in attribute_mapping
                    in_detected = class_id in detected_classes
                    # CRITICAL: Always check pixel count directly, not just if in detected_classes
                    pixel_count = np.sum(segmentation_mask == class_id)
                    attr_name = attribute_mapping.get(class_id, 'NOT_IN_MAPPING')
                    status = "✅" if pixel_count > 0 else "❌"
                    print(f"   {status} class_id={class_id}:")
                    print(f"      - In attribute_mapping: {in_mapping} ({attr_name})")
                    print(f"      - In detected_classes (np.unique): {in_detected}")
                    print(f"      - Pixel count (direct check): {pixel_count}")
                    if pixel_count > 0 and not in_detected:
                        print(f"      ⚠️  WARNING: Has {pixel_count} pixels but NOT in np.unique()!")
                
                print(f"🔍 Checking for nose:")
                for class_id in [10]:  # nose=2
                    in_mapping = class_id in attribute_mapping
                    in_detected = class_id in detected_classes
                    # CRITICAL: Always check pixel count directly, not just if in detected_classes
                    pixel_count = np.sum(segmentation_mask == class_id)
                    attr_name = attribute_mapping.get(class_id, 'NOT_IN_MAPPING')
                    status = "✅" if pixel_count > 0 else "❌"
                    print(f"   {status} class_id={class_id}:")
                    print(f"      - In attribute_mapping: {in_mapping} ({attr_name})")
                    print(f"      - In detected_classes (np.unique): {in_detected}")
                    print(f"      - Pixel count (direct check): {pixel_count}")
                    if pixel_count > 0 and not in_detected:
                        print(f"      ⚠️  WARNING: Has {pixel_count} pixels but NOT in np.unique()!")
                print(f"🔍 ================================================\n")
                
                # Lips mask - ONLY u_lip and l_lip (NOT mouth or nose)
                # Note: 'mouth' (class_id=10) might include too much area
                # Based on CELEBA_ATTRIBUTES: u_lip=11, l_lip=12
                # ALWAYS use direct class_id lookup since we know the exact mapping
                lips_class_ids = []
                
                # First try via attribute_mapping
                for class_id, attr in attribute_mapping.items():
                    if attr in ['u_lip', 'l_lip']:
                        lips_class_ids.append(class_id)
                
                # ALWAYS ALSO check direct class_id (11, 12) regardless of attribute_mapping
                # Check by pixel count, not just detected_classes (np.unique might miss small regions)
                for direct_id in [13, 12]:  # u_lip=11, l_lip=12
                    pixels = np.sum(segmentation_mask == direct_id)  # Check directly, not just in detected_classes
                    if pixels > 0:
                        if direct_id not in lips_class_ids:
                            print(f"   Adding class_id={direct_id} via direct pixel lookup ({pixels} pixels)")
                            lips_class_ids.append(direct_id)
                
                # Remove duplicates and sort
                lips_class_ids = sorted(list(set(lips_class_ids)))
                
                print(f"🔍 Lips class IDs found: {lips_class_ids}")
                if lips_class_ids:
                    # ====================================================================
                    # LOG TRƯỚC KHI TẠO LIPS MASK
                    # ====================================================================
                    print(f"\n📝 ========== TẠO LIPS MASK ==========")
                    print(f"📝 Segmentation mask size (gốc): {segmentation_mask.shape}")
                    for class_id in lips_class_ids:
                        pixels_before = np.sum(segmentation_mask == class_id)
                        attr_name = attribute_mapping.get(class_id, f'class_{class_id}')
                        print(f"   📝 class_id={class_id} ({attr_name}): {pixels_before} pixels TRƯỚC khi tạo mask")
                    
                    # Create mask at ORIGINAL BiSeNet size (512x512) to preserve all pixels
                    lips_mask_full = np.zeros((mask_h, mask_w), dtype=np.float32)
                    total_lips_pixels = 0
                    for class_id in lips_class_ids:
                        pixels = np.sum(segmentation_mask == class_id)
                        attr_name = attribute_mapping.get(class_id, f'class_{class_id}')
                        print(f"   Adding class_id={class_id} ({attr_name}): {pixels} pixels")
                        if pixels > 0:
                            lips_mask_full[segmentation_mask == class_id] = 1.0
                            total_lips_pixels += pixels
                        else:
                            print(f"   ⚠️  Warning: class_id={class_id} has 0 pixels, skipping")
                    
                    # Exclude nose from lips mask (nose is class_id=2)
                    nose_class_id = 10  # Direct class_id for nose
                    nose_pixels_removed = np.sum((lips_mask_full > 0) & (segmentation_mask == nose_class_id))
                    if nose_pixels_removed > 0:
                        lips_mask_full[segmentation_mask == nose_class_id] = 0.0  # Remove nose
                        print(f"   Removed {nose_pixels_removed} nose pixels from lips mask")
                    
                    final_lips_pixels = np.sum(lips_mask_full > 0)
                    if final_lips_pixels > 0:
                        # ====================================================================
                        # LOG TRƯỚC VÀ SAU RESIZE LIPS MASK
                        # ====================================================================
                        print(f"\n🔄 ========== RESIZE LIPS MASK ==========")
                        print(f"🔄 Lips mask TRƯỚC resize: {lips_mask_full.shape}, pixels: {final_lips_pixels}")
                        print(f"🔄 Target size: ({h}, {w})")
                        
                        # Store at original size
                        region_masks_original_size['lips'] = lips_mask_full.copy()
                        # Resize to image size for preview/display only
                        lips_mask_resized = cv2.resize(lips_mask_full, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                        region_masks['lips'] = lips_mask_resized
                        
                        pixels_after_resize = np.sum(lips_mask_resized > 0)
                        print(f"✅ Lips mask SAU resize: {lips_mask_resized.shape}, pixels: {pixels_after_resize}")
                        
                        # So sánh pixel counts
                        expected_pixels = int(final_lips_pixels * (h * w) / (mask_h * mask_w))
                        print(f"📊 Pixel count comparison:")
                        print(f"   - TRƯỚC resize: {final_lips_pixels} pixels at {mask_h}x{mask_w}")
                        print(f"   - SAU resize: {pixels_after_resize} pixels at {h}x{w}")
                        print(f"   - Expected (theoretical): ~{expected_pixels} pixels")
                        if pixels_after_resize < final_lips_pixels * 0.8:
                            print(f"   ⚠️  WARNING: Significant pixel loss detected!")
                        print(f"🔄 ======================================\n")
                        
                        print(f"✅ Lips mask created: {len(lips_class_ids)} class IDs, {total_lips_pixels} pixels before cleanup, {final_lips_pixels} pixels at {mask_h}x{mask_w}, {pixels_after_resize} pixels after resize to {h}x{w}")
                    else:
                        print(f"⚠️  Lips mask empty after processing!")
                else:
                    print(f"⚠️  No lips class IDs found! Available attributes: {list(attribute_mapping.values())}")
                    print(f"   Detected classes: {detected_classes}")
                
                # Eyes mask - ONLY l_eye and r_eye (NOT eye_g or brows)
                # Note: 'eye_g' might include eyebrows, so exclude it
                eyes_class_ids = [class_id for class_id, attr in attribute_mapping.items() 
                                 if attr in ['l_eye', 'r_eye']]  # Removed 'eye_g' to avoid including eyebrows
                if eyes_class_ids:
                    # Create at original mask size
                    eyes_mask_full = np.zeros((mask_h, mask_w), dtype=np.float32)
                    for class_id in eyes_class_ids:
                        eyes_mask_full[segmentation_mask == class_id] = 1.0
                    
                    # Resize to image size
                    eyes_mask_resized = cv2.resize(eyes_mask_full, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                    region_masks['eyes'] = eyes_mask_resized
                    region_masks_original_size['eyes'] = eyes_mask_full.copy()
                    print(f"✅ Eyes mask created with class IDs: {eyes_class_ids}, excluded brows: {brow_class_ids}")
                
                # Eyebrows mask - ONLY l_brow and r_brow
                # Based on CELEBA_ATTRIBUTES: l_brow=6, r_brow=7
                eyebrows_class_ids = []
                
                # First try via attribute_mapping
                for class_id, attr in attribute_mapping.items():
                    if attr in ['l_brow', 'r_brow']:
                        eyebrows_class_ids.append(class_id)
                
                # ALWAYS ALSO check direct class_id (6, 7) regardless of attribute_mapping
                # Check by pixel count, not just detected_classes (np.unique might miss small regions)
                for direct_id in [2, 3]:  # l_brow=6, r_brow=7
                    pixels = np.sum(segmentation_mask == direct_id)  # Check directly, not just in detected_classes
                    if pixels > 0:
                        if direct_id not in eyebrows_class_ids:
                            print(f"   Adding class_id={direct_id} via direct pixel lookup ({pixels} pixels)")
                            eyebrows_class_ids.append(direct_id)
                
                # Remove duplicates and sort
                eyebrows_class_ids = sorted(list(set(eyebrows_class_ids)))
                
                print(f"🔍 Eyebrows class IDs found: {eyebrows_class_ids}")
                if eyebrows_class_ids:
                    # ====================================================================
                    # LOG TRƯỚC KHI TẠO EYEBROWS MASK
                    # ====================================================================
                    print(f"\n📝 ========== TẠO EYEBROWS MASK ==========")
                    print(f"📝 Segmentation mask size (gốc): {segmentation_mask.shape}")
                    for class_id in eyebrows_class_ids:
                        pixels_before = np.sum(segmentation_mask == class_id)
                        attr_name = attribute_mapping.get(class_id, f'class_{class_id}')
                        print(f"   📝 class_id={class_id} ({attr_name}): {pixels_before} pixels TRƯỚC khi tạo mask")
                    
                    # Create at ORIGINAL BiSeNet size (512x512) to preserve all pixels
                    eyebrows_mask_full = np.zeros((mask_h, mask_w), dtype=np.float32)
                    total_brows_pixels = 0
                    for class_id in eyebrows_class_ids:
                        pixels = np.sum(segmentation_mask == class_id)
                        attr_name = attribute_mapping.get(class_id, f'class_{class_id}')
                        print(f"   Adding class_id={class_id} ({attr_name}): {pixels} pixels")
                        if pixels > 0:
                            eyebrows_mask_full[segmentation_mask == class_id] = 1.0
                            total_brows_pixels += pixels
                        else:
                            print(f"   ⚠️  Warning: class_id={class_id} has 0 pixels, skipping")
                    
                    final_brows_pixels = np.sum(eyebrows_mask_full > 0)
                    if final_brows_pixels > 0:
                        # ====================================================================
                        # LOG TRƯỚC VÀ SAU RESIZE EYEBROWS MASK
                        # ====================================================================
                        print(f"\n🔄 ========== RESIZE EYEBROWS MASK ==========")
                        print(f"🔄 Eyebrows mask TRƯỚC resize: {eyebrows_mask_full.shape}, pixels: {final_brows_pixels}")
                        print(f"🔄 Target size: ({h}, {w})")
                        
                        # Store at original size
                        region_masks_original_size['eyebrows'] = eyebrows_mask_full.copy()
                        # Resize to image size for preview/display
                        eyebrows_mask_resized = cv2.resize(eyebrows_mask_full, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                        region_masks['eyebrows'] = eyebrows_mask_resized
                        
                        pixels_after_resize = np.sum(eyebrows_mask_resized > 0)
                        print(f"✅ Eyebrows mask SAU resize: {eyebrows_mask_resized.shape}, pixels: {pixels_after_resize}")
                        
                        # So sánh pixel counts
                        expected_pixels = int(final_brows_pixels * (h * w) / (mask_h * mask_w))
                        print(f"📊 Pixel count comparison:")
                        print(f"   - TRƯỚC resize: {final_brows_pixels} pixels at {mask_h}x{mask_w}")
                        print(f"   - SAU resize: {pixels_after_resize} pixels at {h}x{w}")
                        print(f"   - Expected (theoretical): ~{expected_pixels} pixels")
                        if pixels_after_resize < final_brows_pixels * 0.8:
                            print(f"   ⚠️  WARNING: Significant pixel loss detected!")
                        print(f"🔄 ===========================================\n")
                        
                        print(f"✅ Eyebrows mask created: {len(eyebrows_class_ids)} class IDs, {total_brows_pixels} pixels at {mask_h}x{mask_w}, {pixels_after_resize} pixels after resize to {h}x{w}")
                    else:
                        print(f"⚠️  Eyebrows mask empty after processing!")
                else:
                    print(f"⚠️  No eyebrows detected! Available attributes: {list(attribute_mapping.values())}")
                    print(f"   Detected classes: {detected_classes}")
                
                # Nose mask - for debugging/preview (exclude eyebrows)
                # ALWAYS use direct class_id=2 for nose
                nose_class_id = 10  # Direct class_id for nose
                
                # CRITICAL: Verify class_id mapping before creating mask
                # Check what attribute_mapping says about class_id=2
                mapped_nose_attr = attribute_mapping.get(nose_class_id, 'NOT_IN_MAPPING')
                print(f"🔍 Nose class_id verification: class_id={nose_class_id}, mapped to attribute='{mapped_nose_attr}'")
                
                # Also check r_brow (class_id=7) mapping
                r_brow_class_id = 3
                mapped_r_brow_attr = attribute_mapping.get(r_brow_class_id, 'NOT_IN_MAPPING')
                print(f"🔍 r_brow class_id verification: class_id={r_brow_class_id}, mapped to attribute='{mapped_r_brow_attr}'")
                
                # Check if nose is actually detected - ALWAYS check pixel count directly
                nose_pixels = np.sum(segmentation_mask == nose_class_id)
                nose_in_detected = nose_class_id in detected_classes
                print(f"🔍 Nose detection: class_id={nose_class_id}, pixel_count={nose_pixels}, in_detected={nose_in_detected}")
                if nose_pixels > 0 and not nose_in_detected:
                    print(f"   ⚠️  WARNING: Nose has {nose_pixels} pixels but NOT in np.unique()!")
                
                # Also check r_brow pixels
                r_brow_pixels = np.sum(segmentation_mask == r_brow_class_id)
                print(f"🔍 r_brow detection: class_id={r_brow_class_id}, pixel_count={r_brow_pixels}")
                
                if nose_pixels > 0:
                    # ====================================================================
                    # LOG TRƯỚC KHI TẠO NOSE MASK
                    # ====================================================================
                    print(f"\n📝 ========== TẠO NOSE MASK ==========")
                    print(f"📝 Segmentation mask size (gốc): {segmentation_mask.shape}")
                    print(f"   📝 class_id={nose_class_id} (nose, mapped to '{mapped_nose_attr}'): {nose_pixels} pixels TRƯỚC khi tạo mask")
                    
                    # Create nose mask at original size - ONLY from class_id=2
                    nose_mask_full = np.zeros((mask_h, mask_w), dtype=np.float32)
                    nose_mask_full[segmentation_mask == nose_class_id] = 1.0
                    
                    # CRITICAL VERIFICATION: Check what class_ids are actually in nose_mask_full
                    # Get all pixels where nose_mask_full > 0 and check their class_ids in segmentation_mask
                    nose_mask_indices = np.where(nose_mask_full > 0)
                    if len(nose_mask_indices[0]) > 0:
                        # Sample first 100 pixels to check their class_ids
                        sample_size = min(100, len(nose_mask_indices[0]))
                        sample_indices = (nose_mask_indices[0][:sample_size], nose_mask_indices[1][:sample_size])
                        sample_class_ids = segmentation_mask[sample_indices]
                        unique_in_nose_mask = np.unique(sample_class_ids)
                        print(f"   🔍 Class IDs trong nose_mask (sample {sample_size} pixels): {sorted(unique_in_nose_mask)}")
                        for cid in unique_in_nose_mask:
                            count = np.sum(sample_class_ids == cid)
                            attr_name = attribute_mapping.get(int(cid), f'class_{cid}')
                            print(f"      class_id={cid} ({attr_name}): {count}/{sample_size} pixels in sample")
                    
                    # CRITICAL VERIFICATION AFTER REMOVING EYEBROWS
                    # Check what class_ids are still in nose_mask_full
                    nose_mask_indices_after = np.where(nose_mask_full > 0)
                    if len(nose_mask_indices_after[0]) > 0:
                        sample_size_after = min(100, len(nose_mask_indices_after[0]))
                        sample_indices_after = (nose_mask_indices_after[0][:sample_size_after], nose_mask_indices_after[1][:sample_size_after])
                        sample_class_ids_after = segmentation_mask[sample_indices_after]
                        unique_in_nose_mask_after = np.unique(sample_class_ids_after)
                        print(f"   🔍 Class IDs trong nose_mask SAU KHI LOẠI BỎ EYEBROWS (sample {sample_size_after} pixels): {sorted(unique_in_nose_mask_after)}")
                        for cid in unique_in_nose_mask_after:
                            count_after = np.sum(sample_class_ids_after == cid)
                            attr_name_after = attribute_mapping.get(int(cid), f'class_{cid}')
                            print(f"      class_id={cid} ({attr_name_after}): {count_after}/{sample_size_after} pixels in sample")
                            
                            # WARNING if nose mask contains non-nose class_ids
                            if int(cid) != nose_class_id:
                                print(f"      ⚠️  WARNING: Nose mask contains class_id={cid} ({attr_name_after}), expected only class_id={nose_class_id} (nose)!")
                    
                    # Double-check: verify nose mask only contains nose pixels
                    final_nose_pixels = np.sum(nose_mask_full > 0)
                    final_nose_from_seg = np.sum((segmentation_mask == nose_class_id) & (nose_mask_full > 0))
                    print(f"   Nose mask final: {final_nose_pixels} pixels at {mask_h}x{mask_w} (should match {final_nose_from_seg} nose pixels)")
                    
                    # CRITICAL: Count pixels from each class_id in final nose mask
                    if final_nose_pixels > 0:
                        print(f"   🔍 Chi tiết class_ids trong nose_mask_final:")
                        for cid in range(19):  # Check all possible class_ids 0-18 (18 attributes + background)
                            pixels_in_mask = np.sum((segmentation_mask == cid) & (nose_mask_full > 0))
                            if pixels_in_mask > 0:
                                attr_name_final = attribute_mapping.get(cid, f'class_{cid}')
                                print(f"      class_id={cid:2d} ({attr_name_final:10s}): {pixels_in_mask:6d} pixels")
                                if cid != nose_class_id:
                                    print(f"         ❌ ERROR: Nose mask contains {pixels_in_mask} pixels of class_id={cid} ({attr_name_final}), not nose!")
                    
                    if final_nose_pixels > 0:
                        # ====================================================================
                        # LOG TRƯỚC VÀ SAU RESIZE NOSE MASK
                        # ====================================================================
                        print(f"\n🔄 ========== RESIZE NOSE MASK ==========")
                        print(f"🔄 Nose mask TRƯỚC resize: {nose_mask_full.shape}, pixels: {final_nose_pixels}")
                        print(f"🔄 Target size: ({h}, {w})")
                        
                        # Store at original size
                        region_masks_original_size['nose'] = nose_mask_full.copy()
                        # Resize to image size
                        nose_mask_resized = cv2.resize(nose_mask_full, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                        region_masks['nose'] = nose_mask_resized
                        
                        pixels_after_resize = np.sum(nose_mask_resized > 0)
                        print(f"✅ Nose mask SAU resize: {nose_mask_resized.shape}, pixels: {pixels_after_resize}")
                        
                        # So sánh pixel counts
                        expected_pixels = int(final_nose_pixels * (h * w) / (mask_h * mask_w))
                        print(f"📊 Pixel count comparison:")
                        print(f"   - TRƯỚC resize: {final_nose_pixels} pixels at {mask_h}x{mask_w}")
                        print(f"   - SAU resize: {pixels_after_resize} pixels at {h}x{w}")
                        print(f"   - Expected (theoretical): ~{expected_pixels} pixels")
                        if pixels_after_resize < final_nose_pixels * 0.8:
                            print(f"   ⚠️  WARNING: Significant pixel loss detected!")
                        print(f"🔄 ========================================\n")
                        
                        print(f"✅ Nose mask created: {final_nose_pixels} pixels at {mask_h}x{mask_w}, {pixels_after_resize} pixels after resize to {h}x{w}")
                    else:
                        print(f"⚠️  Nose mask empty after excluding eyebrows, skipping")
                else:
                    print(f"⚠️  Nose not detected (0 pixels), skipping nose mask")
                
                # Debug: Check ALL detected classes and their pixel counts
                print(f"🔍 Full segmentation analysis:")
                print(f"   Detected class IDs in mask: {detected_classes}")
                print(f"   Total pixels in mask: {segmentation_mask.size}")
                
                # Check each important attribute
                important_attrs = ['u_lip', 'l_lip', 'mouth', 'l_eye', 'r_eye', 'eye_g', 'l_brow', 'r_brow', 'nose', 'skin']
                print(f"\n   Pixel counts per attribute:")
                for attr in important_attrs:
                    attr_class_ids = [class_id for class_id, attr_name in attribute_mapping.items() if attr_name == attr]
                    if attr_class_ids:
                        total_pixels = 0
                        for cid in attr_class_ids:
                            pixel_count = np.sum(segmentation_mask == cid)
                            total_pixels += pixel_count
                            if pixel_count > 0:
                                print(f"      {attr} (class_id={cid}): {pixel_count} pixels")
                        if total_pixels == 0:
                            print(f"      {attr}: 0 pixels (NOT detected)")
                    else:
                        print(f"      {attr}: NOT in attribute_mapping!")
                
                # FINAL CHECK: Verify all masks before preview generation
                print(f"\n🔍 FINAL MASK VERIFICATION (before preview generation):")
                print(f"   region_masks keys: {list(region_masks.keys()) if region_masks else 'None'}")
                
                # Check if lips mask has any pixels
                if 'lips' in region_masks:
                    lips_pixel_count = np.sum(region_masks['lips'] > 0)
                    print(f"   ✅ Lips mask: {lips_pixel_count} pixels, shape={region_masks['lips'].shape}")
                else:
                    print(f"   ❌ Lips mask: NOT in region_masks!")
                
                # Check if eyebrows mask has any pixels
                if 'eyebrows' in region_masks:
                    brows_pixel_count = np.sum(region_masks['eyebrows'] > 0)
                    print(f"   ✅ Eyebrows mask: {brows_pixel_count} pixels, shape={region_masks['eyebrows'].shape}")
                else:
                    print(f"   ❌ Eyebrows mask: NOT in region_masks!")
                
                # Check nose mask for eyebrows contamination
                if 'nose' in region_masks:
                    nose_pixel_count = np.sum(region_masks['nose'] > 0)
                    print(f"   ✅ Nose mask: {nose_pixel_count} pixels, shape={region_masks['nose'].shape}")
                    
                    # Also check original-size nose mask
                    if 'region_masks_original_size' in locals() and 'nose' in region_masks_original_size:
                        nose_orig_pixels = np.sum(region_masks_original_size['nose'] > 0)
                        print(f"      Nose mask (original 512x512): {nose_orig_pixels} pixels")
                
                # Skin mask
                skin_class_ids = [class_id for class_id, attr in attribute_mapping.items() 
                                 if attr == 'skin']
                if skin_class_ids:
                    skin_mask_full = np.zeros((mask_h, mask_w), dtype=np.float32)
                    for class_id in skin_class_ids:
                        skin_mask_full[segmentation_mask == class_id] = 1.0
                    
                    skin_mask_resized = cv2.resize(skin_mask_full, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                    region_masks['skin'] = skin_mask_resized
                    region_masks_original_size['skin'] = skin_mask_full.copy()

                # Hair mask
                hair_class_ids = [class_id for class_id, attr in attribute_mapping.items() 
                                  if attr == 'hair']
                if hair_class_ids:
                    hair_mask_full = np.zeros((mask_h, mask_w), dtype=np.float32)
                    for class_id in hair_class_ids:
                        hair_mask_full[segmentation_mask == class_id] = 1.0
                    
                    hair_mask_resized = cv2.resize(hair_mask_full, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                    region_masks['hair'] = hair_mask_resized
                    region_masks_original_size['hair'] = hair_mask_full.copy()
                
                # Create face mask (all non-background regions) at original mask size
                face_mask_original = (segmentation_mask > 0).astype(np.float32)
                # Resize face mask to image size
                face_mask = cv2.resize(face_mask_original, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            else:
                # BiSeNet failed, use rectangular face mask
                padding = int((x2 - x1) * 0.2)
                x1_crop = max(0, x1 - padding)
                y1_crop = max(0, y1 - padding)
                x2_crop = min(w, x2 + padding)
                y2_crop = min(h, y2 + padding)
                face_mask = np.zeros((h, w), dtype=np.float32)
                face_mask[y1_crop:y2_crop, x1_crop:x2_crop] = 1.0
                region_masks = None
        
        # Generate BiSeNet result previews (processed: colorized mask resized and annotated image)
        bisenet_previews = {}
        if bisenet_result is not None:
            # Include raw BiSeNet results if available (before processing)
            if 'raw_bisenet_previews' in locals():
                bisenet_previews.update(raw_bisenet_previews)
                print(f"📊 Added raw BiSeNet previews to response")
            
            # ====================================================================
            # LOG COLORIZED MASK RESIZE PROCESS
            # ====================================================================
            print(f"\n🔄 ========== COLORIZED MASK RESIZE PROCESS ==========")
            
            # Get full image dimensions
            with Image.open(temp_path) as img:
                img_h = img.height
                img_w = img.width
            
            print(f"🔄 Target image size for resize: ({img_h}, {img_w})")
            
            # Use the resized colorized_mask we prepared earlier, or resize now if needed
            if 'colorized_mask_resized_for_preview' in locals():
                print(f"✅ Using pre-resized colorized_mask: {colorized_mask_resized_for_preview.shape}")
                colorized_mask_resized = colorized_mask_resized_for_preview
            elif 'colorized_mask_original' in locals():
                print(f"🔄 Using colorized_mask_original: {colorized_mask_original.shape}")
                # Resize original colorized mask for preview
                if colorized_mask_original.shape[:2] != (img_h, img_w):
                    print(f"   → Resizing from {colorized_mask_original.shape[:2]} to ({img_h}, {img_w})")
                    colorized_mask_resized = cv2.resize(
                        colorized_mask_original,
                        (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    print(f"   ✅ Resized to: {colorized_mask_resized.shape}")
                else:
                    print(f"   ℹ️  Already correct size, no resize needed")
                    colorized_mask_resized = colorized_mask_original
            elif 'colorized_mask' in locals() and colorized_mask is not None:
                print(f"🔄 Using colorized_mask (fallback): {colorized_mask.shape}")
                # Fallback: resize colorized_mask if available
                if colorized_mask.shape[:2] != (img_h, img_w):
                    print(f"   → Resizing from {colorized_mask.shape[:2]} to ({img_h}, {img_w})")
                    colorized_mask_resized = cv2.resize(
                        colorized_mask,
                        (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    print(f"   ✅ Resized to: {colorized_mask_resized.shape}")
                else:
                    print(f"   ℹ️  Already correct size, no resize needed")
                    colorized_mask_resized = colorized_mask
            else:
                print(f"⚠️  No colorized_mask available for resize")
                colorized_mask_resized = None
            
            print(f"✅ Final colorized_mask_resized: {colorized_mask_resized.shape if colorized_mask_resized is not None else 'None'}")
            print(f"🔄 ===================================================\n")
            
            if colorized_mask_resized is not None:
                # Convert resized colorized mask to base64 (for annotated image creation)
                colorized_pil = Image.fromarray(colorized_mask_resized)
                colorized_buffer = BytesIO()
                colorized_pil.save(colorized_buffer, format="PNG")
                colorized_b64 = base64.b64encode(colorized_buffer.getvalue()).decode("utf-8")
                bisenet_previews['colorized_mask'] = f"data:image/png;base64,{colorized_b64}"  # Resized for annotated image
                
                # Create annotated image (blend original with colorized mask)
                img_full = Image.open(temp_path)
                img_rgb_full = np.array(img_full.convert('RGB'))
                
                # Blend
                annotated_image = cv2.addWeighted(img_rgb_full.copy(), 0.6, 
                                                   colorized_mask_resized, 0.4, 0)
                
                # Convert annotated image to base64
                annotated_pil = Image.fromarray(annotated_image)
                annotated_buffer = BytesIO()
                annotated_pil.save(annotated_buffer, format="PNG")
                annotated_b64 = base64.b64encode(annotated_buffer.getvalue()).decode("utf-8")
                bisenet_previews['annotated_image'] = f"data:image/png;base64,{annotated_b64}"
        
        # Generate mask preview images AFTER style application (use styled colors on white background)
        mask_previews = {}
        
        # Apply filter with face mask and region masks
        # STRATEGY: Work at higher resolution (512x512) to preserve small regions
        img = Image.open(temp_path)
        img_rgb_original = np.array(img.convert('RGB'))
        img_h_orig, img_w_orig = img_rgb_original.shape[:2]
        
        # Check if we have original-size masks (from BiSeNet at 512x512)
        if 'region_masks_original_size' in locals() and region_masks_original_size and len(region_masks_original_size) > 0:
            # OPTIMIZED PATH: Use high-resolution masks (512x512) to preserve small regions
            mask_h_actual = list(region_masks_original_size.values())[0].shape[0]
            mask_w_actual = list(region_masks_original_size.values())[0].shape[1]
            
            # Resize image to match mask size (512x512) for better quality
            # This ensures small regions like eyebrows are preserved
            img_rgb_highres = cv2.resize(img_rgb_original, (mask_w_actual, mask_h_actual), interpolation=cv2.INTER_LINEAR)
            print(f"🔄 Resizing image from {img_w_orig}x{img_h_orig} to {mask_w_actual}x{mask_h_actual} for filter application")
            
            # Apply filter at high resolution using original-size masks
            face_mask_for_filter = face_mask_original if 'face_mask_original' in locals() else None
            if face_mask_for_filter is None and 'segmentation_mask' in locals():
                face_mask_for_filter = (segmentation_mask > 0).astype(np.float32)
            
            filtered_highres = apply_style_to_image(
                img_rgb_highres,
                style_data,
                intensity=intensity,
                face_mask=face_mask_for_filter,
                use_regions=True,
                region_masks=region_masks_original_size
            )
            
            # Resize filtered result back to original image size
            filtered = cv2.resize(filtered_highres, (img_w_orig, img_h_orig), interpolation=cv2.INTER_LINEAR)
            print(f"🔄 Resizing filtered image back to {img_w_orig}x{img_h_orig}")
        else:
            # FALLBACK: Use resized masks (if BiSeNet failed or no original-size masks)
            print(f"⚠️  Using resized masks (fallback path)")
            filtered = apply_style_to_image(
                img_rgb_original,
                style_data,
                intensity=intensity,
                face_mask=face_mask if 'face_mask' in locals() else None,
                use_regions=(region_masks is not None and len(region_masks) > 0),
                region_masks=region_masks if 'region_masks' in locals() else None
            )
        
        # Now that we have the styled image, build region previews using STYLED colors
        if region_masks:
            print(f"\n🔍 ========== MASK PREVIEW GENERATION (STYLED COLORS) ==========")
            print(f"🔍 Generating mask previews for {len(region_masks)} regions: {list(region_masks.keys())}")
            styled_img = filtered
            white_bg_img = np.ones_like(styled_img, dtype=np.uint8) * 255
            for region_name, region_mask in region_masks.items():
                mask_pixel_count = np.sum(region_mask > 0)
                if mask_pixel_count == 0:
                    print(f"   ⚠️  {region_name} mask has 0 pixels, skip")
                    continue
                mask_bool = (region_mask > 0.5)
                mask_bool_3 = np.repeat(mask_bool[:, :, None], 3, axis=2)
                region_preview = white_bg_img.copy()
                region_preview[mask_bool_3] = styled_img[mask_bool_3]
                mask_pil = Image.fromarray(region_preview)
                mask_buffer = BytesIO()
                mask_pil.save(mask_buffer, format="PNG")
                mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")
                mask_previews[region_name] = f"data:image/png;base64,{mask_b64}"
        
        # Save filtered image with high quality to preserve sharpness
        filtered_img = Image.fromarray(filtered)
        filtered_filename = f"filtered_{uuid.uuid4().hex[:8]}.jpg"
        filtered_path = os.path.join("/tmp", filtered_filename)
        # Use quality=98 and optimize=False to preserve image sharpness
        filtered_img.save(filtered_path, quality=98, optimize=False)
        
        # Read filtered image
        with open(filtered_path, "rb") as f:
            filtered_data = f.read()
        
        # Encode to base64
        filtered_b64 = base64.b64encode(filtered_data).decode("utf-8")
        
        return {
            "success": True,
            "style_id": style_id,
            "intensity": intensity,
            "filtered_image": f"data:image/jpeg;base64,{filtered_b64}",
            "original_size": Image.open(temp_path).size,
            "bisenet_previews": bisenet_previews,  # BiSeNet raw results (colorized mask, annotated image)
            "mask_previews": mask_previews,  # Individual region masks
            "regions_detected": list(region_masks.keys()) if region_masks else [],
            "attribute_mapping": attribute_mapping if 'attribute_mapping' in locals() else {}
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error applying filter: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply filter: {str(e)}"
        )
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if 'filtered_path' in locals() and os.path.exists(filtered_path) and filtered_path != temp_path:
            os.remove(filtered_path)


@router.get("/style/{style_id}")
async def get_style_info(style_id: str):
    """
    Get style information including download URLs
    """
    style_data = get_style(style_id)
    if not style_data:
        raise HTTPException(status_code=404, detail=f"Style {style_id} not found")
    
    return style_data


@router.post("/style/create_default")
async def create_default_style_endpoint():
    """
    Create a default red lipstick filter for testing
    """
    from app.services.default_filter import create_default_red_lips_style
    
    try:
        style_data = create_default_red_lips_style()
        return style_data
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error creating default style: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create default style: {str(e)}"
        )


@router.get("/styles")
async def list_styles_endpoint(limit: int = 20, offset: int = 0):
    """
    List all available styles
    """
    result = list_styles(limit=limit, offset=offset)
    
    # Load full style data for each style
    full_styles = []
    for style_entry in result["styles"]:
        style_id = style_entry["style_id"]
        full_data = get_style(style_id)
        if full_data:
            full_styles.append(full_data)
    
    return {
        "styles": full_styles,
        "total": result["total"],
        "limit": limit,
        "offset": offset
    }
