"""
WebGL Shader Generation Service
Generates WebGL shader code for real-time makeup filter application.

Following Phase 1, Step 5 from makeup_filter_system.md
"""

from typing import Dict, List, Any
import os


def generate_fragment_shader(
    style_data: Dict[str, Any],
    use_luts: bool = True
) -> str:
    """
    Generate WebGL fragment shader for makeup application
    
    Args:
        style_data: Complete style data with regions
        use_luts: Whether to use 3D LUTs (True) or direct color mixing (False)
    
    Returns:
        GLSL fragment shader code as string
    """
    if use_luts:
        return _generate_lut_based_shader(style_data)
    else:
        return _generate_direct_color_shader(style_data)


def _generate_lut_based_shader(style_data: Dict[str, Any]) -> str:
    """Generate shader using 3D LUTs"""
    shader = """
#version 300 es
precision highp float;

// Input texture (camera frame)
uniform sampler2D u_texture;
uniform sampler2D u_lut_lips;
uniform sampler2D u_lut_eyes;
uniform sampler2D u_lut_skin;
uniform sampler2D u_lut_eyebrows;
uniform sampler2D u_lut_cheeks;

// Region masks
uniform sampler2D u_mask_lips;
uniform sampler2D u_mask_eyes;
uniform sampler2D u_mask_skin;
uniform sampler2D u_mask_eyebrows;
uniform sampler2D u_mask_cheeks;
uniform sampler2D u_mask_face;  // Face mask to only apply filter to face region

// Intensity controls
uniform float u_intensity;
uniform float u_lips_intensity;
uniform float u_eyes_intensity;
uniform float u_skin_intensity;
uniform float u_eyebrows_intensity;
uniform float u_cheeks_intensity;

// Varying from vertex shader
in vec2 v_texCoord;
out vec4 fragColor;

// 3D LUT lookup function (using 2D texture representation)
vec3 lut_lookup(sampler2D lut, vec3 color) {
    float lut_size = 32.0;
    float scale = (lut_size - 1.0) / lut_size;
    float offset = 1.0 / (2.0 * lut_size);
    
    // Map RGB to 2D texture coordinates
    vec3 scaled = color * scale + offset;
    float b = scaled.b;
    float g = scaled.g;
    
    // Calculate row and column in 2D representation
    float row = floor(b * lut_size);
    float col = g * lut_size;
    
    vec2 lut_coord = vec2(col / lut_size + scaled.r / lut_size, row / lut_size);
    return texture(lut, lut_coord).rgb;
}

void main() {
    vec4 original = texture(u_texture, v_texCoord);
    vec3 color = original.rgb;
    
    // Sample face mask - STRICT: only apply filter to face region, not background
    float mask_face = texture(u_mask_face, v_texCoord).r;
    float face_threshold = 0.15; // Higher threshold to ensure background is completely untouched
    
    // If no face detected (background), return original pixel unchanged - NO FILTER AT ALL
    // This check happens BEFORE any filter processing, so background is 100% unaffected
    if (mask_face < face_threshold) {
        fragColor = original;
        return;
    }
    
    // Sample region masks (only within face region)
    float mask_lips = texture(u_mask_lips, v_texCoord).r;
    float mask_eyes = texture(u_mask_eyes, v_texCoord).r;
    float mask_skin = texture(u_mask_skin, v_texCoord).r;
    float mask_eyebrows = texture(u_mask_eyebrows, v_texCoord).r;
    float mask_cheeks = texture(u_mask_cheeks, v_texCoord).r;
    
    // Start with original color - will blend filtered colors into it
    vec3 final_color = color;
    
    // Apply LUTs to each region based on masks
    // Intensity controls how much of the filtered color is blended in (0 = no filter, 1 = full filter)
    // Only apply within face region and where region mask exists
    
    // Lips
    if (mask_lips > 0.05) {
        vec3 lips_filtered = lut_lookup(u_lut_lips, color);
        float blend_alpha = mask_lips * u_lips_intensity * u_intensity;
        final_color = mix(final_color, lips_filtered, blend_alpha);
    }
    
    // Eyes
    if (mask_eyes > 0.05) {
        vec3 eyes_filtered = lut_lookup(u_lut_eyes, color);
        float blend_alpha = mask_eyes * u_eyes_intensity * u_intensity;
        final_color = mix(final_color, eyes_filtered, blend_alpha);
    }
    
    // Skin
    if (mask_skin > 0.05) {
        vec3 skin_filtered = lut_lookup(u_lut_skin, color);
        float blend_alpha = mask_skin * u_skin_intensity * u_intensity;
        final_color = mix(final_color, skin_filtered, blend_alpha);
    }
    
    // Eyebrows
    if (mask_eyebrows > 0.05) {
        vec3 eyebrows_filtered = lut_lookup(u_lut_eyebrows, color);
        float blend_alpha = mask_eyebrows * u_eyebrows_intensity * u_intensity;
        final_color = mix(final_color, eyebrows_filtered, blend_alpha);
    }
    
    // Cheeks
    if (mask_cheeks > 0.05) {
        vec3 cheeks_filtered = lut_lookup(u_lut_cheeks, color);
        float blend_alpha = mask_cheeks * u_cheeks_intensity * u_intensity;
        final_color = mix(final_color, cheeks_filtered, blend_alpha);
    }
    
    fragColor = vec4(final_color, original.a);
}
"""
    return shader


def _generate_direct_color_shader(style_data: Dict[str, Any]) -> str:
    """Generate shader using direct color mixing (fallback)"""
    shader = """
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform sampler2D u_mask_lips;
uniform sampler2D u_mask_eyes;
uniform sampler2D u_mask_skin;

// Color uniforms (RGB 0-1)
uniform vec3 u_lips_color;
uniform vec3 u_eyes_color;
uniform vec3 u_skin_color;

uniform float u_intensity;
uniform float u_lips_intensity;
uniform float u_eyes_intensity;
uniform float u_skin_intensity;

in vec2 v_texCoord;
out vec4 fragColor;

void main() {
    vec4 original = texture(u_texture, v_texCoord);
    vec3 color = original.rgb;
    
    float mask_lips = texture(u_mask_lips, v_texCoord).r;
    float mask_eyes = texture(u_mask_eyes, v_texCoord).r;
    float mask_skin = texture(u_mask_skin, v_texCoord).r;
    
    // Simple color mixing
    color = mix(color, u_lips_color, mask_lips * u_lips_intensity * u_intensity);
    color = mix(color, u_eyes_color, mask_eyes * u_eyes_intensity * u_intensity);
    color = mix(color, u_skin_color, mask_skin * u_skin_intensity * u_intensity);
    
    fragColor = vec4(color, original.a);
}
"""
    return shader


def generate_vertex_shader() -> str:
    """Generate simple pass-through vertex shader"""
    shader = """
#version 300 es
in vec4 a_position;
in vec2 a_texCoord;

out vec2 v_texCoord;

void main() {
    gl_Position = a_position;
    v_texCoord = a_texCoord;
}
"""
    return shader


def save_shader(code: str, filepath: str):
    """Save shader code to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(code)


def generate_all_shaders(
    style_data: Dict[str, Any],
    output_dir: str,
    use_luts: bool = True
) -> Dict[str, str]:
    """
    Generate all shader files for a style
    
    Args:
        style_data: Complete style data
        output_dir: Directory to save shaders
        use_luts: Whether to use LUT-based shaders
    
    Returns:
        Dictionary with shader file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    style_id = style_data.get('style_id', 'unknown')
    
    shader_files = {}
    
    # Generate fragment shader
    frag_shader = generate_fragment_shader(style_data, use_luts)
    frag_path = os.path.join(output_dir, f"{style_id}_fragment.glsl")
    save_shader(frag_shader, frag_path)
    shader_files['fragment'] = frag_path
    
    # Generate vertex shader
    vert_shader = generate_vertex_shader()
    vert_path = os.path.join(output_dir, f"{style_id}_vertex.glsl")
    save_shader(vert_shader, vert_path)
    shader_files['vertex'] = vert_path
    
    return shader_files

