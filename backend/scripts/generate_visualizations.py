"""
Script to generate visualizations for project report
Usage: python scripts/generate_visualizations.py <style_id> [output_dir]
"""

import sys
import os
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.style_storage import get_style
from app.services.storage import MinioService
from app.utils.visualization import (
    create_color_palette_visualization,
    create_combined_visualization,
    extract_colors_from_style_data
)
import io
from PIL import Image


async def generate_visualizations(style_id: str, output_dir: str = "visualizations"):
    """Generate all visualizations for a style"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Generating visualizations for style: {style_id}")
    
    # Get style
    style_data = get_style(style_id)
    if not style_data:
        print(f"❌ Style {style_id} not found")
        return
    
    print(f"✅ Found style: {style_data.get('name', style_id)}")
    
    # Extract colors
    print("🎨 Extracting color palettes...")
    region_colors, region_weights = extract_colors_from_style_data(style_data)
    
    if not region_colors:
        print("⚠️  No color data found. Style may need to be recreated.")
        return
    
    # Generate color palette visualization
    print("📸 Creating color palette visualization...")
    palette_img = create_color_palette_visualization(
        region_colors,
        region_weights,
        output_path=os.path.join(output_dir, f"{style_id}_color_palette.png"),
        size=(1200, 800)
    )
    print(f"✅ Saved: {output_dir}/{style_id}_color_palette.png")
    
    # Load region overlays and save previews
    storage_service = MinioService()
    download_urls = style_data.get('download_urls', {})
    region_mask_urls = download_urls.get('region_masks', {})
    
    region_overlays: Dict[str, Image.Image] = {}
    for region_name, mask_url in region_mask_urls.items():
        print(f"📥 Loading region overlay for {region_name}...")
        try:
            from urllib.parse import urlparse, unquote
            if mask_url.startswith('/api/'):
                object_name = mask_url.replace('/api/makeup/storage/file/', '')
                object_name = unquote(object_name)
            else:
                parsed = urlparse(mask_url)
                object_name = parsed.path.lstrip('/')
                if 'facetory-storage' in object_name:
                    object_name = object_name.split('facetory-storage/', 1)[1]
            
            overlay_data = await storage_service.download_file(object_name)
            overlay_img = Image.open(io.BytesIO(overlay_data)).convert('RGBA')
            region_overlays[region_name] = overlay_img

            overlay_path = os.path.join(output_dir, f"{style_id}_overlay_{region_name}.png")
            overlay_img.save(overlay_path)
            print(f"✅ Saved overlay: {overlay_path}")
        except Exception as e:
            print(f"⚠️  Could not load overlay for {region_name}: {e}")
    
    # Generate combined visualization
    if region_colors and region_overlays:
        print("📸 Creating combined visualization...")
        combined_img = create_combined_visualization(
            region_colors,
            region_weights,
            region_overlays,
            output_path=os.path.join(output_dir, f"{style_id}_combined.png"),
            size=(1600, 1200)
        )
        print(f"✅ Saved: {output_dir}/{style_id}_combined.png")
    
    print(f"\n✨ All visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_visualizations.py <style_id> [output_dir]")
        sys.exit(1)
    
    style_id = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualizations"
    
    asyncio.run(generate_visualizations(style_id, output_dir))

