"""
Visualization utilities for makeup filter system
Generates visualizations for:
- Dominant color palettes from segmented regions
- LUT visualizations
- For use in project reports and documentation
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import rgb2hex
import io


def lab_to_rgb(lab_color: List[float]) -> Tuple[int, int, int]:
    """Convert LAB color to RGB"""
    lab_array = np.array([[lab_color]], dtype=np.float32)
    rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)
    rgb = rgb_array[0, 0]
    return tuple((rgb * 255).astype(int).clip(0, 255))


def create_color_palette_visualization(
    region_colors: Dict[str, List[Tuple[int, int, int]]],
    region_weights: Dict[str, List[float]],
    output_path: Optional[str] = None,
    size: Tuple[int, int] = (800, 600)
) -> Image.Image:
    """
    Create a visualization of dominant color palettes for each region
    
    Args:
        region_colors: Dict mapping region names to list of RGB colors
        region_weights: Dict mapping region names to list of color weights
        output_path: Optional path to save image
        size: Image size (width, height)
    
    Returns:
        PIL Image with color palette visualization
    """
    regions = list(region_colors.keys())
    n_regions = len(regions)
    
    # Create image
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Layout parameters
    margin = 50
    region_height = (size[1] - margin * 2) // n_regions
    color_box_size = 80
    spacing = 20
    
    y_pos = margin
    
    for region_name in regions:
        colors = region_colors[region_name]
        weights = region_weights.get(region_name, [1.0] * len(colors))
        
        # Draw region name
        draw.text((margin, y_pos + region_height // 2 - 10), 
                 region_name.upper(), fill='black', font=font)
        
        # Draw color boxes
        x_pos = margin + 150
        for i, (color, weight) in enumerate(zip(colors, weights)):
            # Draw color box
            box = [x_pos, y_pos + 10, x_pos + color_box_size, y_pos + 10 + color_box_size]
            draw.rectangle(box, fill=color)
            
            # Draw border
            draw.rectangle(box, outline='gray', width=2)
            
            # Draw weight percentage
            weight_text = f"{weight:.1%}"
            bbox = draw.textbbox((0, 0), weight_text, font=font_small)
            text_width = bbox[2] - bbox[0]
            text_x = x_pos + (color_box_size - text_width) // 2
            draw.text((text_x, y_pos + color_box_size + 5), 
                     weight_text, fill='gray', font=font_small)
            
            # Draw RGB values
            rgb_text = f"RGB{color}"
            bbox = draw.textbbox((0, 0), rgb_text, font=font_small)
            text_width = bbox[2] - bbox[0]
            text_x = x_pos + (color_box_size - text_width) // 2
            draw.text((text_x, y_pos + color_box_size + 20), 
                     rgb_text, fill='darkgray', font=font_small)
            
            x_pos += color_box_size + spacing
        
        y_pos += region_height
    
    # Draw title
    title = "Dominant Color Palettes - Extracted from Segmented Facial Regions"
    bbox = draw.textbbox((0, 0), title, font=font)
    title_width = bbox[2] - bbox[0]
    title_x = (size[0] - title_width) // 2
    draw.text((title_x, 10), title, fill='black', font=font)
    
    if output_path:
        img.save(output_path)
    
    return img


def visualize_lut_2d(
    lut: np.ndarray,
    region_name: str,
    output_path: Optional[str] = None,
    size: Tuple[int, int] = (600, 600)
) -> Image.Image:
    """
    Visualize a 3D LUT as 2D slices
    
    Args:
        lut: 3D LUT array (should be 32x32x32x3 or similar)
        region_name: Name of the region (e.g., 'lips', 'eyes')
        output_path: Optional path to save image
        size: Image size
    
    Returns:
        PIL Image with LUT visualization
    """
    lut_size = lut.shape[0]  # Assuming cubic LUT
    
    # Create a grid showing multiple slices of the LUT
    # Show slices at different B values (blue channel)
    n_slices = 4
    slice_indices = [int(lut_size * i / (n_slices + 1)) for i in range(1, n_slices + 1)]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'3D LUT Visualization - {region_name.upper()} Region', 
                 fontsize=16, fontweight='bold')
    
    for idx, (ax, slice_idx) in enumerate(zip(axes.flat, slice_indices)):
        # Extract 2D slice (R-G plane at fixed B value)
        slice_2d = lut[slice_idx, :, :, :]  # Shape: (lut_size, lut_size, 3)
        
        # Convert to uint8 if needed
        if slice_2d.max() <= 1.0:
            slice_2d = (slice_2d * 255).astype(np.uint8)
        else:
            slice_2d = slice_2d.astype(np.uint8)
        
        # Display slice
        ax.imshow(slice_2d)
        ax.set_title(f'B = {slice_idx}/{lut_size}', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img = img.resize(size, Image.Resampling.LANCZOS)
    plt.close()
    
    if output_path:
        img.save(output_path)
    
    return img


def visualize_lut_color_transformation(
    lut: np.ndarray,
    region_name: str,
    output_path: Optional[str] = None,
    size: Tuple[int, int] = (800, 400)
) -> Image.Image:
    """
    Visualize LUT by showing original color grid and transformed color grid
    
    Args:
        lut: 3D LUT array
        region_name: Name of the region
        output_path: Optional path to save image
        size: Image size
    
    Returns:
        PIL Image showing color transformation
    """
    lut_size = lut.shape[0]
    
    # Create original color grid
    n_samples = 8
    original_colors = []
    transformed_colors = []
    
    for r in range(n_samples):
        for g in range(n_samples):
            for b in range(n_samples):
                r_idx = int(r * (lut_size - 1) / (n_samples - 1))
                g_idx = int(g * (lut_size - 1) / (n_samples - 1))
                b_idx = int(b * (lut_size - 1) / (n_samples - 1))
                
                # Original color (normalized)
                orig_rgb = [r_idx / (lut_size - 1), g_idx / (lut_size - 1), b_idx / (lut_size - 1)]
                original_colors.append(orig_rgb)
                
                # Transformed color from LUT
                if lut.ndim == 4:
                    trans_rgb = lut[b_idx, g_idx, r_idx]  # Note: LUT indexing might be different
                    if trans_rgb.max() <= 1.0:
                        trans_rgb = trans_rgb
                    else:
                        trans_rgb = trans_rgb / 255.0
                    transformed_colors.append(trans_rgb)
                else:
                    transformed_colors.append(orig_rgb)  # Fallback
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'LUT Color Transformation - {region_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Original colors
    orig_array = np.array(original_colors).reshape(n_samples, n_samples, n_samples, 3)
    # Show a 2D slice
    orig_slice = orig_array[:, :, n_samples // 2, :]
    ax1.imshow(orig_slice)
    ax1.set_title('Original Color Grid', fontsize=12)
    ax1.axis('off')
    
    # Transformed colors
    trans_array = np.array(transformed_colors).reshape(n_samples, n_samples, n_samples, 3)
    trans_slice = trans_array[:, :, n_samples // 2, :]
    ax2.imshow(trans_slice)
    ax2.set_title('Transformed Color Grid (via LUT)', fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img = img.resize(size, Image.Resampling.LANCZOS)
    plt.close()
    
    if output_path:
        img.save(output_path)
    
    return img


def create_combined_visualization(
    region_colors: Dict[str, List[Tuple[int, int, int]]],
    region_weights: Dict[str, List[float]],
    region_overlays: Dict[str, Image.Image],
    output_path: Optional[str],
    size: Tuple[int, int] = (1600, 1200)
) -> Image.Image:
    """
    Create a combined visualization showing color palettes and RGBA region overlays.

    Args:
        region_colors: Dict mapping region names to list of RGB colors
        region_weights: Dict mapping region names to list of color weights
        region_overlays: Dict mapping region names to PIL Images (RGBA overlays)
        output_path: Path to save combined image (optional)
        size: Image size
    """
    # Create color palette visualization
    palette_img = create_color_palette_visualization(
        region_colors, region_weights, size=(size[0], size[1] // 2)
    ).convert('RGBA')

    combined = Image.new('RGBA', size, color='white')
    combined.paste(palette_img, (0, 0))

    if region_overlays:
        overlay_regions = list(region_overlays.keys())
        columns = max(1, len(overlay_regions))
        overlay_width = size[0] // columns
        overlay_height = size[1] // 2 - 20

        for idx, region_name in enumerate(overlay_regions):
            overlay_img = region_overlays[region_name].convert('RGBA')
            overlay_img = overlay_img.resize(
                (overlay_width - 20, overlay_height),
                Image.Resampling.LANCZOS
            )

            x = idx * overlay_width + 10
            y = size[1] // 2 + 10

            # Create label bar
            label_height = 30
            label_img = Image.new('RGBA', (overlay_width - 20, label_height), (255, 255, 255, 220))
            draw = ImageDraw.Draw(label_img)
            draw.text((10, 5), region_name.upper(), fill=(0, 0, 0, 255))

            combined.paste(overlay_img, (x, y), overlay_img)
            combined.paste(label_img, (x, y + overlay_img.height + 5), label_img)

    if output_path:
        combined.convert('RGB').save(output_path)

    return combined.convert('RGB')


def extract_colors_from_style_data(style_data: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """
    Extract color palettes from style data structure
    
    Returns:
        (region_colors, region_weights) dictionaries
    """
    region_colors = {}
    region_weights = {}
    
    style_params = style_data.get('style_parameters', {})
    
    for region_name, region_data in style_params.items():
        colors = []
        weights = []
        
        # Extract primary color
        if 'primary_lab' in region_data and region_data['primary_lab']:
            primary_rgb = lab_to_rgb(region_data['primary_lab'])
            colors.append(primary_rgb)
            weights.append(region_data.get('coverage_intensity', 0.8))
        
        # Extract secondary color
        if 'secondary_lab' in region_data and region_data.get('secondary_lab'):
            secondary_rgb = lab_to_rgb(region_data['secondary_lab'])
            colors.append(secondary_rgb)
            weights.append(1.0 - region_data.get('coverage_intensity', 0.8))
        
        # Extract color weights if available
        if 'color_weights' in region_data:
            weights = region_data['color_weights']
        
        if colors:
            region_colors[region_name] = colors
            region_weights[region_name] = weights
    
    return region_colors, region_weights

