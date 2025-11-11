"""
Style Extraction Service
Extracts makeup style parameters from segmented facial regions using:
- LAB color space conversion
- K-means clustering
- Histogram analysis
Following the flow in makeup_filter_system.md
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Any
from collections import Counter

# Mapping from CelebAMask-HQ attributes to makeup regions
REGION_MAPPING = {
    'u_lip': 'lips',
    'l_lip': 'lips',
    'mouth': 'lips',
    'l_eye': 'eyes',
    'r_eye': 'eyes',
    'eye_g': 'eyes',
    'l_brow': 'eyebrows',
    'r_brow': 'eyebrows',
    'skin': 'skin',
    'nose': 'nose',
    # Note: cheeks might need special handling as they're not directly in CelebAMask-HQ
}

def rgb_to_lab(rgb_image: np.ndarray) -> np.ndarray:
    """Convert RGB image to LAB color space"""
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # OpenCV uses BGR, so convert RGB to BGR first
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    else:
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    
    return lab_image


def extract_region_pixels(image: np.ndarray, mask: np.ndarray, region_class_id: int) -> np.ndarray:
    """Extract pixels for a specific region from the segmentation mask"""
    region_mask = (mask == region_class_id)
    if not np.any(region_mask):
        return np.array([]).reshape(0, 3)
    
    region_pixels = image[region_mask]
    return region_pixels


def kmeans_color_extraction(pixels: np.ndarray, k: int = 3, max_iter: int = 300) -> Dict[str, Any]:
    """
    Extract dominant colors using K-means clustering
    Returns: dominant colors, their weights, and cluster labels
    """
    if len(pixels) == 0:
        return {
            'colors': [],
            'weights': [],
            'labels': None
        }
    
    if len(pixels) < k:
        k = len(pixels)
    
    # Reshape pixels for KMeans (expects 2D array)
    if len(pixels.shape) == 1:
        pixels = pixels.reshape(-1, 1)
    
    # Run K-means
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=max_iter, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Get cluster centers (dominant colors)
    colors = kmeans.cluster_centers_
    
    # Calculate weights (proportion of pixels in each cluster)
    label_counts = Counter(labels)
    total_pixels = len(labels)
    weights = [label_counts[i] / total_pixels for i in range(k)]
    
    # Sort by weight (descending)
    sorted_indices = sorted(range(k), key=lambda i: weights[i], reverse=True)
    colors = colors[sorted_indices]
    weights = [weights[i] for i in sorted_indices]
    
    return {
        'colors': colors.astype(int).tolist(),
        'weights': weights,
        'labels': labels
    }


def analyze_histogram_distribution(pixels: np.ndarray, bins: int = 32) -> Dict[str, Any]:
    """Analyze color distribution using histogram"""
    if len(pixels) == 0:
        return {
            'intensity_distribution': [],
            'variance': 0,
            'mean': 0
        }
    
    # Calculate histogram for each channel (L, A, B)
    histograms = []
    for channel in range(min(3, pixels.shape[1])):
        hist, _ = np.histogram(pixels[:, channel], bins=bins, range=(0, 255))
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        histograms.append(hist.tolist())
    
    # Calculate statistics
    variance = float(np.var(pixels))
    mean = float(np.mean(pixels))
    
    # Intensity distribution (using L channel if LAB, otherwise first channel)
    if len(histograms) > 0:
        intensity_distribution = histograms[0]
    else:
        intensity_distribution = []
    
    return {
        'intensity_distribution': intensity_distribution,
        'variance': variance,
        'mean': mean,
        'channel_histograms': histograms
    }


def calculate_coverage(mask: np.ndarray, region_class_id: int, image_shape: Tuple[int, int]) -> float:
    """Calculate coverage percentage of a region in the face"""
    region_mask = (mask == region_class_id)
    region_pixels = np.sum(region_mask)
    total_face_pixels = np.sum(mask > 0)  # All non-background pixels
    
    if total_face_pixels == 0:
        return 0.0
    
    coverage = region_pixels / total_face_pixels
    return float(coverage)


def extract_blend_softness(pixels: np.ndarray, labels: np.ndarray = None) -> float:
    """Estimate blend softness based on color variance and transitions"""
    if len(pixels) == 0:
        return 0.0
    
    # Higher variance = softer blend (more transitions)
    # Lower variance = harder blend (more uniform)
    variance = float(np.var(pixels))
    
    # Normalize to 0-1 range (rough estimate, may need tuning)
    softness = min(1.0, variance / 100.0)
    return softness


def extract_texture_type(pixels: np.ndarray) -> str:
    """Determine texture type (matte, glossy, etc.) based on color characteristics"""
    if len(pixels) == 0:
        return "unknown"
    
    # Convert to LAB if needed and analyze L channel (lightness)
    if pixels.shape[1] >= 3:
        lightness = pixels[:, 0] if len(pixels.shape) > 1 else pixels
        variance = float(np.var(lightness))
        mean_lightness = float(np.mean(lightness))
        
        # High variance + high lightness = glossy
        # Low variance + medium lightness = matte
        if variance > 500 and mean_lightness > 150:
            return "glossy"
        elif variance < 200:
            return "matte"
        else:
            return "satin"
    
    return "matte"


def extract_style_for_region(
    image_rgb: np.ndarray,
    image_lab: np.ndarray,
    mask: np.ndarray,
    region_class_id: int,
    region_name: str,
    k_clusters: int = 3
) -> Dict[str, Any]:
    """
    Extract style parameters for a single facial region
    
    Args:
        image_rgb: Original RGB image
        image_lab: LAB converted image
        mask: Segmentation mask
        region_class_id: Class ID of the region in the mask
        region_name: Name of the region
        k_clusters: Number of clusters for K-means
    
    Returns:
        Dictionary with style parameters
    """
    # Extract pixels for this region
    pixels_rgb = extract_region_pixels(image_rgb, mask, region_class_id)
    pixels_lab = extract_region_pixels(image_lab, mask, region_class_id)
    
    if len(pixels_lab) == 0:
        return {
            'region': region_name,
            'primary_lab': None,
            'secondary_lab': None,
            'coverage_intensity': 0.0,
            'blend_softness': 0.0,
            'texture_type': 'unknown',
            'color_weights': [],
            'intensity_distribution': [],
            'average_rgb': None
        }
    
    # K-means clustering in LAB space
    kmeans_result = kmeans_color_extraction(pixels_lab, k=k_clusters)
    
    # Get primary and secondary colors
    colors_lab = kmeans_result['colors']
    weights = kmeans_result['weights']
    
    primary_lab = colors_lab[0] if len(colors_lab) > 0 else None
    secondary_lab = colors_lab[1] if len(colors_lab) > 1 else None
    
    # Histogram analysis
    hist_analysis = analyze_histogram_distribution(pixels_lab)
    
    # Coverage intensity
    coverage = calculate_coverage(mask, region_class_id, image_rgb.shape[:2])
    
    # Blend softness
    softness = extract_blend_softness(pixels_lab, kmeans_result.get('labels'))
    
    # Texture type
    texture = extract_texture_type(pixels_lab)
    
    avg_rgb = pixels_rgb.mean(axis=0).astype(int).tolist() if len(pixels_rgb) > 0 else None

    return {
        'region': region_name,
        'primary_lab': primary_lab,
        'secondary_lab': secondary_lab,
        'coverage_intensity': coverage,
        'blend_softness': softness,
        'texture_type': texture,
        'color_weights': weights,
        'intensity_distribution': hist_analysis['intensity_distribution'],
        'variance': hist_analysis['variance'],
        'mean_lab': hist_analysis['mean'],
        'average_rgb': avg_rgb
    }


def aggregate_regions_to_makeup_areas(region_styles: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Aggregate individual region styles into makeup areas
    E.g., combine u_lip + l_lip into 'lips'
    """
    makeup_areas = {
        'lips': [],
        'eyes': [],
        'eyebrows': [],
        'skin': [],
        'nose': []
    }
    
    # Map regions to makeup areas
    for region_name, style_data in region_styles.items():
        mapped_area = REGION_MAPPING.get(region_name)
        if mapped_area and mapped_area in makeup_areas:
            makeup_areas[mapped_area].append({
                'region': region_name,
                **style_data
            })
    
    # Aggregate styles for each makeup area
    result = {}
    
    for area_name, regions in makeup_areas.items():
        if not regions:
            continue
        
        if area_name == 'lips':
            # Combine lip regions
            if regions:
                # Get maximum coverage, but ensure minimum of 0.6 for visibility
                coverage = max([r.get('coverage_intensity', 0) for r in regions])
                # Normalize coverage: if too low, use at least 0.6
                # Coverage intensity represents pixel coverage, but for makeup application,
                # we want stronger effect even if region is small
                normalized_coverage = max(coverage, 0.6) if coverage < 0.3 else coverage
                
                result['lips'] = {
                    'primary_lab': regions[0].get('primary_lab'),
                    'secondary_lab': regions[0].get('secondary_lab'),
                    'coverage_intensity': normalized_coverage,
                    'blend_softness': float(np.mean([r.get('blend_softness', 0) for r in regions])),
                    'texture_type': regions[0].get('texture_type', 'matte'),
                    'average_rgb': regions[0].get('average_rgb')
                }
        
        elif area_name == 'eyes':
            # Eyeshadow colors from multiple regions
            if regions:
                eyeshadow_colors = []
                for r in regions:
                    if r.get('primary_lab'):
                        eyeshadow_colors.append(r['primary_lab'])
                
                result['eyes'] = {
                    'eyeshadow_colors_lab': eyeshadow_colors[:3],  # Top 3 colors
                    'gradient_direction': 'vertical',  # Default, can be enhanced
                    'intensity_curve': [0.2, 0.8, 0.6, 0.3],  # Default curve
                    'eyeliner_thickness': 2,  # Default
                    'blend_pattern': 'soft_transition',
                    'coverage_intensity': max([r.get('coverage_intensity', 0) for r in regions]),
                    'average_rgb': regions[0].get('average_rgb')
                }
        
        elif area_name == 'eyebrows':
            if regions:
                result['eyebrows'] = {
                    'color_lab': regions[0].get('primary_lab'),
                    'coverage_intensity': max([r.get('coverage_intensity', 0) for r in regions]),
                    'average_rgb': regions[0].get('average_rgb')
                }
        
        elif area_name == 'skin':
            # Skin/foundation adjustments
            if regions:
                skin_style = regions[0]
                result['skin'] = {
                    'foundation_adjustment': [0, 0, 0],  # Placeholder, can be enhanced
                    'smoothing_level': float(1.0 - skin_style.get('blend_softness', 0.5)),
                    'highlight_zones': [],  # Can be enhanced with spatial analysis
                    'base_color_lab': skin_style.get('primary_lab'),
                    'average_rgb': skin_style.get('average_rgb')
                }
        
        elif area_name == 'nose':
            if regions:
                result['nose'] = {
                    'primary_lab': regions[0].get('primary_lab'),
                    'coverage_intensity': max([r.get('coverage_intensity', 0) for r in regions]),
                    'average_rgb': regions[0].get('average_rgb')
                }
    
    return result


def extract_makeup_style(
    image_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    attribute_mapping: Dict[int, str]
) -> Dict[str, Any]:
    """
    Main function to extract makeup style from segmented image
    
    Args:
        image_rgb: RGB image (numpy array)
        segmentation_mask: Segmentation mask with class IDs (numpy array)
        attribute_mapping: Mapping from class_id to attribute name
            e.g., {1: 'skin', 10: 'mouth', 11: 'u_lip', ...}
    
    Returns:
        Dictionary with complete style parameters
    """
    # Convert to LAB color space
    image_lab = rgb_to_lab(image_rgb)
    
    # Resize mask to match image if needed
    if segmentation_mask.shape[:2] != image_rgb.shape[:2]:
        segmentation_mask = cv2.resize(
            segmentation_mask, 
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Extract style for each region
    region_styles = {}
    
    for class_id, attr_name in attribute_mapping.items():
        if class_id == 0:  # Skip background
            continue
        
        style = extract_style_for_region(
            image_rgb=image_rgb,
            image_lab=image_lab,
            mask=segmentation_mask,
            region_class_id=class_id,
            region_name=attr_name,
            k_clusters=3
        )
        
        region_styles[attr_name] = style
    
    # Aggregate into makeup areas
    makeup_style = aggregate_regions_to_makeup_areas(region_styles)
    
    # Generate style ID (simple hash, can be enhanced)
    import hashlib
    style_hash = hashlib.md5(str(makeup_style).encode()).hexdigest()[:8]
    style_id = f"style_{style_hash}"
    
    return {
        'style_id': style_id,
        'regions': region_styles,
        **makeup_style  # Unpack lips, eyes, eyebrows, skin
    }

