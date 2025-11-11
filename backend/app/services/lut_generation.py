"""
LUT (Look-Up Table) Generation Service
Generates 3D color lookup tables from style parameters for real-time makeup application.

Following Phase 1, Step 5 from makeup_filter_system.md
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
from pathlib import Path
import os


def lab_to_rgb(lab: List[float]) -> List[int]:
    """Convert LAB color to RGB"""
    if lab is None or len(lab) != 3:
        return [0, 0, 0]
    
    # OpenCV LAB format: L (0-100), A (0-255), B (0-255)
    # But extracted LAB might be in different format or scaled
    # Normalize L to 0-100 range if it's out of bounds
    lab_normalized = list(lab)
    
    # If L is out of range (should be 0-100), normalize it
    if lab[0] > 100:
        # Assume L was scaled to 0-255, normalize to 0-100
        lab_normalized[0] = lab[0] * 100.0 / 255.0
    elif lab[0] < 0:
        lab_normalized[0] = 0.0
    
    # Clamp L to valid range
    lab_normalized[0] = np.clip(lab_normalized[0], 0, 100)
    # Clamp A and B to valid range (0-255)
    lab_normalized[1] = np.clip(lab[1], 0, 255)
    lab_normalized[2] = np.clip(lab[2], 0, 255)
    
    # Convert single LAB value to numpy array
    lab_array = np.array([[lab_normalized]], dtype=np.float32)
    
    # Convert LAB to BGR (OpenCV uses BGR)
    bgr = cv2.cvtColor(lab_array, cv2.COLOR_LAB2BGR)
    
    # Convert to RGB and clip to valid range
    rgb = bgr[0][0][::-1]  # Reverse BGR to RGB
    rgb = np.clip(rgb * 255, 0, 255).astype(int).tolist()
    
    return rgb


def generate_3d_lut(
    primary_lab: List[float] = None,
    secondary_lab: List[float] = None,
    coverage_intensity: float = 1.0,
    blend_softness: float = 0.5,
    texture_type: str = "matte",
    lut_size: int = 32  # Standard size for 3D LUT (32x32x32 is common)
) -> np.ndarray:
    """
    Generate a 3D color lookup table from style parameters
    
    Args:
        primary_lab: Primary color in LAB space [L, A, B]
        secondary_lab: Secondary color in LAB space [L, A, B]
        coverage_intensity: How strongly to apply the color (0-1)
        blend_softness: How softly to blend (0-1)
        texture_type: Type of texture (matte, satin, glossy)
        lut_size: Size of each dimension in the 3D LUT (default 32)
    
    Returns:
        3D numpy array of shape (lut_size, lut_size, lut_size, 3) containing RGB values
    """
    # Initialize 3D LUT
    lut = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.uint8)
    
    # Default neutral colors if no colors provided
    if primary_lab is None:
        primary_rgb = [128, 128, 128]
    else:
        primary_rgb = lab_to_rgb(primary_lab)
    
    if secondary_lab is None:
        secondary_rgb = primary_rgb
    else:
        secondary_rgb = lab_to_rgb(secondary_lab)
    
    # Convert RGB to normalized [0, 1]
    primary_norm = np.array(primary_rgb) / 255.0
    secondary_norm = np.array(secondary_rgb) / 255.0
    
    # Generate LUT by interpolating between original color and makeup color
    for r in range(lut_size):
        for g in range(lut_size):
            for b in range(lut_size):
                # Normalize input RGB to [0, 1]
                input_rgb = np.array([r, g, b]) / (lut_size - 1)
                
                # Calculate distance from input to neutral (for blend calculation)
                # Use primary color as the "makeup" target
                makeup_color = primary_norm
                
                # Interpolate between original and makeup color based on coverage
                # For makeup, we want to shift colors toward the makeup color
                # Coverage intensity should represent how strongly to apply the color
                # For very low coverage (like 0.029), we need to amplify it or use a minimum threshold
                
                # Normalize coverage to ensure minimum visibility
                # If coverage is too low, use a minimum threshold (0.5) or amplify it
                effective_coverage = max(coverage_intensity, 0.5) if coverage_intensity < 0.1 else coverage_intensity
                
                # Calculate how close input color is to the makeup target
                # The closer the input is to makeup color, the stronger the transformation
                color_distance = np.linalg.norm(input_rgb - makeup_color)
                max_distance = np.sqrt(3.0)  # Maximum distance in RGB cube
                similarity = 1.0 - (color_distance / max_distance)
                
                # Blend factor: stronger for similar colors, modulated by coverage
                blend_factor = effective_coverage * (0.3 + 0.7 * similarity)  # 30-100% based on similarity
                blend_factor = np.clip(blend_factor, 0, 1)
                
                # Apply color transformation: shift toward makeup color
                output_rgb = input_rgb * (1 - blend_factor) + makeup_color * blend_factor
                
                # Apply texture effect
                if texture_type == "glossy":
                    # Increase brightness for glossy
                    output_rgb = output_rgb * 1.1
                elif texture_type == "matte":
                    # Slight desaturation for matte
                    gray = np.mean(output_rgb)
                    output_rgb = output_rgb * 0.9 + gray * 0.1
                
                # Clip and convert to uint8
                output_rgb = np.clip(output_rgb, 0, 1)
                lut[r, g, b] = (output_rgb * 255).astype(np.uint8)
    
    return lut


def save_lut_binary(lut: np.ndarray, filepath: str):
    """Save 3D LUT as binary file"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Flatten and save as binary
    with open(filepath, 'wb') as f:
        # Write shape info first
        shape = lut.shape
        f.write(np.array(shape, dtype=np.int32).tobytes())
        # Write data
        f.write(lut.tobytes())


def load_lut_binary(filepath_or_bytes) -> np.ndarray:
    """Load 3D LUT from binary file or BytesIO"""
    if isinstance(filepath_or_bytes, str):
        with open(filepath_or_bytes, 'rb') as f:
            data = _read_lut_from_file(f)
    else:
        # Assume it's a file-like object (BytesIO)
        filepath_or_bytes.seek(0)  # Reset to beginning
        data = _read_lut_from_file(filepath_or_bytes)
    return data


def _read_lut_from_file(file_obj) -> np.ndarray:
    """Read LUT data from file object"""
    # Read shape (4 int32 values: [R, G, B, channels])
    shape_bytes = file_obj.read(4 * 4)
    shape = np.frombuffer(shape_bytes, dtype=np.int32)
    # Read data
    data = np.frombuffer(file_obj.read(), dtype=np.uint8)
    # Reshape
    lut = data.reshape(shape)
    return lut


def generate_lut_for_region(
    region_style: Dict[str, Any],
    lut_size: int = 32
) -> np.ndarray:
    """
    Generate LUT for a specific makeup region
    
    Args:
        region_style: Style parameters for the region
        lut_size: Size of LUT (default 32)
    
    Returns:
        3D LUT array
    """
    primary_lab = region_style.get('primary_lab')
    secondary_lab = region_style.get('secondary_lab')
    coverage = region_style.get('coverage_intensity', 0.8)
    softness = region_style.get('blend_softness', 0.6)
    texture = region_style.get('texture_type', 'matte')
    
    return generate_3d_lut(
        primary_lab=primary_lab,
        secondary_lab=secondary_lab,
        coverage_intensity=coverage,
        blend_softness=softness,
        texture_type=texture,
        lut_size=lut_size
    )


def generate_all_luts(
    style_data: Dict[str, Any],
    output_dir: str,
    lut_size: int = 32
) -> Dict[str, str]:
    """
    Generate LUTs for all makeup regions
    
    Args:
        style_data: Complete style data from extract_makeup_style
        output_dir: Directory to save LUT files
        lut_size: Size of LUTs
    
    Returns:
        Dictionary mapping region names to LUT file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    style_id = style_data.get('style_id', 'unknown')
    lut_files = {}
    
    # Regions to generate LUTs for
    regions = ['lips', 'eyes', 'eyebrows', 'skin', 'cheeks']
    
    for region in regions:
        if region in style_data:
            region_style = style_data[region]
            
            # Generate LUT
            lut = generate_lut_for_region(region_style, lut_size)
            
            # Save LUT
            lut_filename = f"{style_id}_{region}_lut.bin"
            lut_path = os.path.join(output_dir, lut_filename)
            save_lut_binary(lut, lut_path)
            
            lut_files[region] = lut_path
    
    return lut_files

