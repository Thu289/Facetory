"""
API endpoints for visualization generation
Provides endpoints to generate visualizations for project reports
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from typing import Optional, Dict, Any
import io
import base64
from PIL import Image, ImageDraw

from app.services.storage import MinioService
from app.services.style_storage import get_style
from app.utils.visualization import (
    create_color_palette_visualization,
    create_combined_visualization,
    extract_colors_from_style_data,
)

router = APIRouter()
storage_service = MinioService()


@router.get("/style/{style_id}/color-palette")
async def get_color_palette_visualization(
    style_id: str,
    width: int = Query(800, ge=100, le=2000),
    height: int = Query(600, ge=100, le=2000)
):
    """
    Generate color palette visualization for a style
    
    Returns PNG image showing dominant colors extracted from each region
    """
    try:
        # Get style from storage
        style_data = get_style(style_id)
        if not style_data:
            raise HTTPException(status_code=404, detail="Style not found")
        
        # Extract colors from style data
        region_colors, region_weights = extract_colors_from_style_data(style_data)
        
        if not region_colors:
            raise HTTPException(
                status_code=404, 
                detail="No color data found in style. Style may need to be recreated."
            )
        
        # Create visualization
        img = create_color_palette_visualization(
            region_colors,
            region_weights,
            size=(width, height)
        )
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=color_palette_{style_id}.png"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")


@router.get("/style/{style_id}/lut-visualization/{region}")
async def get_lut_visualization(
    style_id: str,
    region: str,
    visualization_type: str = Query("overlay", regex="^(overlay|2d|transformation)$"),
    width: int = Query(800, ge=100, le=2000),
    height: int = Query(400, ge=100, le=2000)
):
    """
    Return RGBA overlay visualization for a specific region.
    """
    try:
        style_data = get_style(style_id)
        if not style_data:
            raise HTTPException(status_code=404, detail="Style not found")

        download_urls = style_data.get('download_urls', {})
        region_masks = download_urls.get('region_masks', {})

        if region not in region_masks:
            raise HTTPException(
                status_code=404,
                detail=f"Region mask not found for region: {region}"
            )

        mask_url = region_masks[region]
        from urllib.parse import urlparse, unquote
        if mask_url.startswith('/api/'):
            object_name = mask_url.replace('/api/makeup/storage/file/', '')
            object_name = unquote(object_name)
        else:
            parsed = urlparse(mask_url)
            object_name = parsed.path.lstrip('/')
            if 'facetory-storage' in object_name:
                object_name = object_name.split('facetory-storage/', 1)[1]

        try:
            overlay_bytes = await storage_service.download_file(object_name)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading region mask: {str(e)}"
            )

        overlay_img = Image.open(io.BytesIO(overlay_bytes)).convert('RGBA')
        overlay_img = overlay_img.resize((width, height), Image.Resampling.LANCZOS)

        img_bytes = io.BytesIO()
        overlay_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=overlay_{region}_{style_id}.png"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")


@router.get("/style/{style_id}/combined-visualization")
async def get_combined_visualization(
    style_id: str,
    width: int = Query(1600, ge=800, le=3000),
    height: int = Query(1200, ge=600, le=3000)
):
    """
    Generate combined visualization showing both color palettes and LUTs
    """
    try:
        # Get style from storage
        style_data = get_style(style_id)
        if not style_data:
            raise HTTPException(status_code=404, detail="Style not found")
        
        # Extract colors
        region_colors, region_weights = extract_colors_from_style_data(style_data)
        
        download_urls = style_data.get('download_urls', {})
        region_mask_urls = download_urls.get('region_masks', {})
        region_overlays: Dict[str, Image.Image] = {}

        from urllib.parse import urlparse, unquote
        for region_name, mask_url in region_mask_urls.items():
            try:
                if mask_url.startswith('/api/'):
                    object_name = mask_url.replace('/api/makeup/storage/file/', '')
                    object_name = unquote(object_name)
                else:
                    parsed = urlparse(mask_url)
                    object_name = parsed.path.lstrip('/')
                    if 'facetory-storage' in object_name:
                        object_name = object_name.split('facetory-storage/', 1)[1]

                overlay_bytes = await storage_service.download_file(object_name)
                overlay_img = Image.open(io.BytesIO(overlay_bytes)).convert('RGBA')
                region_overlays[region_name] = overlay_img
            except Exception as e:
                print(f"Warning: Could not load region overlay for {region_name}: {e}")
        
        if not region_colors and not region_overlays:
            raise HTTPException(
                status_code=404,
                detail="No visualization data available. Style may need to be recreated."
            )
        
        # Create combined visualization
        img = create_combined_visualization(
            region_colors,
            region_weights,
            region_overlays,
            output_path=None,  # Will return image instead
            size=(width, height)
        )
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=combined_visualization_{style_id}.png"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")


@router.get("/style/{style_id}/mask-visualization")
async def get_mask_visualization(
    style_id: str,
    _include_luts: bool = Query(False),
    width: int = Query(2000, ge=800, le=3000),
    height: int = Query(1500, ge=600, le=3000)
):
    """
    Generate visualization showing region masks from segmentation and LUT transformations
    
    Shows:
    - Column 1: Region masks (from segmentation)
    - Column 2: LUT visualization for each region
    - Column 3: Combined visualization (mask + LUT application)
    """
    try:
        style_data = get_style(style_id)
        if not style_data:
            raise HTTPException(status_code=404, detail="Style not found")
        
        mask_previews = style_data.get('metadata', {}).get('mask_previews', {})
        if not mask_previews:
            mask_previews = style_data.get('mask_previews', {})
        
        if not mask_previews:
            raise HTTPException(
                status_code=404,
                detail="No mask previews found in style. Style may need to be recreated."
            )

        download_urls = style_data.get('download_urls', {})
        region_mask_urls = download_urls.get('region_masks', {})

        from urllib.parse import urlparse, unquote
        region_overlays: Dict[str, Image.Image] = {}
        for region_name, mask_url in region_mask_urls.items():
            try:
                if mask_url.startswith('/api/'):
                    object_name = mask_url.replace('/api/makeup/storage/file/', '')
                    object_name = unquote(object_name)
                else:
                    parsed = urlparse(mask_url)
                    object_name = parsed.path.lstrip('/')
                    if 'facetory-storage' in object_name:
                        object_name = object_name.split('facetory-storage/', 1)[1]

                overlay_bytes = await storage_service.download_file(object_name)
                overlay_img = Image.open(io.BytesIO(overlay_bytes)).convert('RGBA')
                region_overlays[region_name] = overlay_img
            except Exception as e:
                print(f"Warning: Could not load overlay for {region_name}: {e}")

        margin = 20
        regions = list(mask_previews.keys())
        num_regions = len(regions)
        row_height = max(150, (height - margin * (num_regions + 1)) // max(1, num_regions))
        canvas_height = row_height * num_regions + margin * (num_regions + 1)
        canvas = Image.new('RGBA', (width, canvas_height), color='white')
        draw = ImageDraw.Draw(canvas)
        column_width = (width - margin * 4) // 3

        for idx, region_name in enumerate(regions):
            y_offset = margin + idx * (row_height + margin)
            preview_data = mask_previews[region_name]
            try:
                if preview_data.startswith('data:image'):
                    _, encoded = preview_data.split(',', 1)
                    mask_img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGBA')
                else:
                    mask_img = Image.open(io.BytesIO(base64.b64decode(preview_data))).convert('RGBA')
            except Exception:
                continue

            mask_img = mask_img.resize((column_width, row_height), Image.Resampling.LANCZOS)
            canvas.paste(mask_img, (margin, y_offset), mask_img)

            overlay_img = region_overlays.get(region_name)
            if overlay_img:
                overlay_resized = overlay_img.resize((column_width, row_height), Image.Resampling.LANCZOS)
                canvas.paste(overlay_resized, (margin * 2 + column_width, y_offset), overlay_resized)

                combined_img = overlay_resized.copy()
                canvas.paste(
                    combined_img,
                    (margin * 3 + column_width * 2, y_offset),
                    combined_img
                )

            draw.text((margin, y_offset - 18 if y_offset - 18 > 0 else y_offset + 5),
                      region_name.upper(), fill=(0, 0, 0, 255))

        canvas = canvas.crop((0, 0, width, min(canvas_height, height)))

        img_bytes = io.BytesIO()
        canvas.convert('RGB').save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=mask_visualization_{style_id}.png"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")


@router.get("/style/{style_id}/visualization-data")
async def get_visualization_data(style_id: str):
    """
    Get raw visualization data (colors, weights) as JSON
    Useful for custom visualizations
    """
    try:
        style_data = get_style(style_id)
        if not style_data:
            raise HTTPException(status_code=404, detail="Style not found")
        region_colors, region_weights = extract_colors_from_style_data(style_data)
        
        # Convert RGB tuples to lists for JSON serialization
        colors_json = {
            region: [list(color) for color in colors]
            for region, colors in region_colors.items()
        }
        
        return {
            "style_id": style_id,
            "region_colors": colors_json,
            "region_weights": region_weights,
            "regions": list(region_colors.keys())
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data: {str(e)}")

