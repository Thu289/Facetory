"""
Mask visualization utilities
Shows region masks from segmentation and how they're used with LUTs
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Any, Optional
import io
import base64


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 image string to numpy array"""
    if base64_str.startswith('data:image'):
        # Remove data URL prefix
        base64_str = base64_str.split(',', 1)[1]
    
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)


def create_mask_lut_visualization(
    mask_previews: Dict[str, str],
    luts: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
    size: Tuple[int, int] = (2000, 1500)
) -> Image.Image:
    """
    Create visualization showing region masks and their corresponding LUT transformations
    
    Args:
        mask_previews: Dict mapping region names to base64-encoded mask preview images
        luts: Dict mapping region names to LUT arrays
        output_path: Optional path to save image
        size: Image size (width, height)
    
    Returns:
        PIL Image showing masks and LUTs side by side
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Filter regions that have both mask and LUT
    regions = [r for r in mask_previews.keys() if r in luts]
    
    if not regions:
        # If no matching regions, just show masks
        regions = list(mask_previews.keys())[:6]  # Max 6 regions
    
    n_regions = len(regions)
    if n_regions == 0:
        raise ValueError("No regions to visualize")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    gs = gridspec.GridSpec(n_regions, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Region Masks from Segmentation & LUT Transformations', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for idx, region_name in enumerate(regions):
        # Column 1: Original mask preview
        ax1 = fig.add_subplot(gs[idx, 0])
        if region_name in mask_previews:
            mask_img = decode_base64_image(mask_previews[region_name])
            ax1.imshow(mask_img)
            ax1.set_title(f'{region_name.upper()}\nMask Preview', fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, f'No mask for\n{region_name}', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f'{region_name.upper()}\nMask Preview', fontsize=12)
        ax1.axis('off')
        
        # Column 2: LUT visualization (if available)
        ax2 = fig.add_subplot(gs[idx, 1])
        if region_name in luts:
            lut = luts[region_name]
            # Show 2D slice of LUT
            lut_size = lut.shape[0]
            slice_idx = lut_size // 2
            slice_2d = lut[slice_idx, :, :, :]
            if slice_2d.max() <= 1.0:
                slice_2d = (slice_2d * 255).astype(np.uint8)
            else:
                slice_2d = slice_2d.astype(np.uint8)
            ax2.imshow(slice_2d)
            ax2.set_title(f'{region_name.upper()}\nLUT (B={slice_idx}/{lut_size})', 
                         fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, f'No LUT for\n{region_name}', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'{region_name.upper()}\nLUT', fontsize=12)
        ax2.axis('off')
        
        # Column 3: Combined visualization (mask + LUT effect)
        ax3 = fig.add_subplot(gs[idx, 2])
        if region_name in mask_previews and region_name in luts:
            # Show conceptual visualization: mask overlay on LUT
            mask_img = decode_base64_image(mask_previews[region_name])
            # Create a visualization showing how mask selects LUT regions
            # Simple: show mask with colored overlay indicating LUT areas
            combined = mask_img.copy()
            # Add a subtle colored border to indicate LUT application
            mask_bool = np.sum(mask_img, axis=2) < 255 * 3  # Non-white pixels
            if np.any(mask_bool):
                # Create a colored overlay
                overlay = combined.copy()
                overlay[mask_bool] = [255, 200, 200]  # Light red tint
                combined = cv2.addWeighted(combined, 0.7, overlay, 0.3, 0)
            ax3.imshow(combined)
            ax3.set_title(f'{region_name.upper()}\nMask + LUT Application', 
                         fontsize=12, fontweight='bold')
        elif region_name in mask_previews:
            mask_img = decode_base64_image(mask_previews[region_name])
            ax3.imshow(mask_img)
            ax3.set_title(f'{region_name.upper()}\nMask Only', fontsize=12)
        else:
            ax3.text(0.5, 0.5, f'No data for\n{region_name}', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f'{region_name.upper()}\nCombined', fontsize=12)
        ax3.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
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


def create_simple_mask_grid(
    mask_previews: Dict[str, str],
    output_path: Optional[str] = None,
    size: Tuple[int, int] = (1600, 1200),
    cols: int = 3
) -> Image.Image:
    """
    Create a simple grid showing all region masks
    
    Args:
        mask_previews: Dict mapping region names to base64-encoded mask preview images
        output_path: Optional path to save image
        size: Image size (width, height)
        cols: Number of columns in grid
    
    Returns:
        PIL Image with mask grid
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    regions = list(mask_previews.keys())
    n_regions = len(regions)
    
    if n_regions == 0:
        raise ValueError("No masks to visualize")
    
    rows = (n_regions + cols - 1) // cols
    
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Region Masks from Segmentation', fontsize=16, fontweight='bold', y=0.98)
    
    for idx, region_name in enumerate(regions):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])
        
        mask_img = decode_base64_image(mask_previews[region_name])
        ax.imshow(mask_img)
        ax.set_title(region_name.upper(), fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
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

