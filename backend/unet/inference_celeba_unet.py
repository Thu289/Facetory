import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import os
import base64
import io

# Import the UNet model and attributes from training script
from train_celeba_unet import UNet, CELEBA_ATTRIBUTES

# Color palette for visualization (20 colors for 19 attributes + background)
PALETTE = [
    [0, 0, 0],        # 0: background - black
    [255, 0, 0],      # 1: skin - red
    [0, 255, 0],      # 2: nose - green
    [0, 0, 255],      # 3: eye_g - blue
    [255, 255, 0],    # 4: l_eye - yellow
    [255, 0, 255],    # 5: r_eye - magenta
    [0, 255, 255],    # 6: l_brow - cyan
    [128, 0, 0],      # 7: r_brow - dark red
    [0, 128, 0],      # 8: l_ear - dark green
    [0, 0, 128],      # 9: r_ear - dark blue
    [128, 128, 0],    # 10: mouth - olive
    [128, 0, 128],    # 11: u_lip - purple
    [0, 128, 128],    # 12: l_lip - teal
    [255, 128, 0],    # 13: hair - orange
    [128, 255, 0],    # 14: hat - lime
    [0, 255, 128],    # 15: ear_r - spring green
    [128, 0, 255],    # 16: neck - violet
    [255, 0, 128],    # 17: neck_l - pink
    [0, 128, 255],    # 18: cloth - sky blue
]

def load_model(checkpoint_path):
    """Load the trained U-Net model"""
    model = UNet(in_channels=3, out_channels=len(CELEBA_ATTRIBUTES) + 1)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained model.")
    
    model.eval()
    return model

def preprocess_image(image, target_size=(512, 512)):
    """Preprocess image for model input"""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor and add batch dimension
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image

def predict_mask(model, image, device='cpu'):
    """Predict segmentation mask for the input image"""
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        mask = torch.argmax(output, dim=1)
    
    return mask.cpu().numpy()[0]  # Remove batch dimension

def colorize_mask(mask):
    """Convert segmentation mask to colored image"""
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for class_id in range(len(PALETTE)):
        colored_mask[mask == class_id] = PALETTE[class_id]
    
    return colored_mask

def extract_region_colors(image, mask):
    """Extract average colors for each facial region"""
    region_colors = {}
    
    # Convert image to RGB if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    else:
        image_rgb = np.array(image)
    
    # Resize image to match mask size
    image_rgb = cv2.resize(image_rgb, (mask.shape[1], mask.shape[0]))
    
    for i, attr_name in enumerate(CELEBA_ATTRIBUTES):
        class_id = i + 1  # +1 because 0 is background
        region_mask = (mask == class_id)
        
        if np.any(region_mask):
            # Get pixels for this region
            region_pixels = image_rgb[region_mask]
            
            # Calculate average color
            avg_color = np.mean(region_pixels, axis=0).astype(int)
            region_colors[attr_name] = {
                'rgb': avg_color.tolist(),
                'hex': '#{:02x}{:02x}{:02x}'.format(avg_color[0], avg_color[1], avg_color[2])
            }
        else:
            # Region not found in mask
            region_colors[attr_name] = {
                'rgb': [0, 0, 0],
                'hex': '#000000'
            }
    
    return region_colors

def create_annotated_image(image, mask):
    """Create an annotated image showing the segmentation overlay"""
    # Convert image to RGB if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    else:
        image_rgb = np.array(image)
    
    # Resize image to match mask size
    image_rgb = cv2.resize(image_rgb, (mask.shape[1], mask.shape[0]))
    
    # Create colored mask
    colored_mask = colorize_mask(mask)
    
    # Create overlay (blend original image with colored mask)
    overlay = cv2.addWeighted(image_rgb, 0.7, colored_mask, 0.3, 0)
    
    return overlay

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    # Convert to PIL Image
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def process_image_with_celeba_unet(image_path, checkpoint_path='best_celeba_unet.pth', device='cpu'):
    """Main function to process an image with CelebAMask-HQ U-Net"""
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Load and preprocess image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    # Preprocess image
    input_tensor = preprocess_image(image)
    
    # Predict mask
    mask = predict_mask(model, input_tensor, device)
    
    # Colorize mask
    colorized_mask = colorize_mask(mask)
    
    # Extract region colors
    region_colors = extract_region_colors(image, mask)
    
    # Create annotated image
    annotated_image = create_annotated_image(image, mask)
    
    # Convert to base64
    colorized_mask_b64 = image_to_base64(colorized_mask)
    annotated_image_b64 = image_to_base64(annotated_image)
    
    return {
        'mask': mask.tolist(),
        'colorized_mask': colorized_mask_b64,
        'annotated_image': annotated_image_b64,
        'region_colors': region_colors,
        'attributes': CELEBA_ATTRIBUTES
    }

def main():
    """Test the CelebAMask-HQ U-Net inference"""
    # Example usage
    test_image_path = "backend/data/CelebAMask-HQ/CelebA-HQ-img/29999.jpg"
    
    if os.path.exists(test_image_path):
        print(f"Processing test image: {test_image_path}")
        result = process_image_with_celeba_unet(test_image_path)
        
        print("Extracted region colors:")
        for attr, color_info in result['region_colors'].items():
            print(f"  {attr}: {color_info['hex']} (RGB: {color_info['rgb']})")
        
        print(f"Number of attributes: {len(result['attributes'])}")
        print("Processing completed!")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please ensure the CelebAMask-HQ dataset is properly set up.")

if __name__ == "__main__":
    main() 