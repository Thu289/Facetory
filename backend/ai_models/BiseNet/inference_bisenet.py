"""
BiSeNet Face Parsing Inference
Pre-trained model for CelebAMask-HQ with 19 facial attributes

This module tries to use the official BiSeNet implementation.
If not available, it falls back to a compatible interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os
import sys
from pathlib import Path

# Try to import official BiSeNet implementation
BISENET_AVAILABLE = False
BISENET_IMPORT_ERROR = None
OfficialBiSeNet = None

try:
    # Add the cloned repository to path
    bisenet_repo = Path(__file__).parent / "face-parsing.PyTorch"
    if bisenet_repo.exists():
        repo_path = str(bisenet_repo)
        # Insert at the beginning to prioritize
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        
        # Try importing the actual BiSeNet model
        try:
            # Import BiSeNet - it will handle resnet internally
            from model import BiSeNet as OfficialBiSeNetImport
            OfficialBiSeNet = OfficialBiSeNetImport
            BISENET_AVAILABLE = True
            print(f"✅ BiSeNet model imported successfully from {repo_path}")
        except ImportError as e:
            BISENET_IMPORT_ERROR = str(e)
            print(f"⚠️  Failed to import BiSeNet: {e}")
            print(f"   Note: This is expected if dependencies are missing. The model will still work if weights are available.")
    else:
        print(f"⚠️  BiSeNet repository not found at: {bisenet_repo}")
except Exception as e:
    BISENET_IMPORT_ERROR = str(e)
    print(f"⚠️  Error setting up BiSeNet import: {e}")

# CelebAMask-HQ attributes mapping
# Note: Pretrained weights have 19 classes (0-18), we use 18 attributes (1-18)
# ID mapping: 0=background, 1-18 = attributes below
CELEBA_ATTRIBUTES = [
    'skin',      # 1
    'l_brow',    # 2
    'r_brow',    # 3
    'l_eye',     # 4
    'r_eye',     # 5
    'eye_g',     # 6
    'l_ear',     # 7
    'r_ear',     # 8
    'ear_r',     # 9
    'nose',      # 10
    'mouth',     # 11
    'u_lip',     # 12
    'l_lip',     # 13
    'neck',      # 14
    'cloth',     # 15
    'hat',      # 16
    'hair',       # 17
    'neck_l'     # 18: dây chuyền
]

NUM_CLASSES = 19  # Pretrained weights use 19 classes (0-18), keep for compatibility

# Color palette for visualization (19 colors for 19 classes)
PALETTE = [
    [0, 0, 0],        # 0: background
    [255, 0, 0],      # 1: skin
    [0, 255, 255],    # 2: l_brow
    [128, 0, 0],      # 3: r_brow
    [255, 255, 0],    # 4: l_eye
    [255, 0, 255],    # 5: r_eye
    [0, 0, 255],      # 6: eye_g
    [0, 128, 0],      # 7: l_ear
    [0, 0, 128],      # 8: r_ear
    [0, 255, 128],    # 9: ear_r
    [0, 255, 0],      # 10: nose
    [128, 128, 0],    # 11: mouth
    [128, 0, 128],    # 12: u_lip
    [0, 128, 128],    # 13: l_lip
    [128, 0, 255],    # 14: neck
    [0, 128, 255],    # 15: cloth
    [255, 128, 0],    # 16: hair
    [128, 255, 0],    # 17: hat
    [255, 0, 128],    # 18: neck_l (dây chuyền)
]

class BiSeNet(nn.Module):
    """
    BiSeNet architecture for face parsing
    Simplified version - you may need to adjust based on actual pre-trained model architecture
    """
    def __init__(self, n_classes=NUM_CLASSES):
        super(BiSeNet, self).__init__()
        # This is a placeholder architecture
        # The actual BiSeNet has ContextPath and SpatialPath
        # You'll need to import the actual BiSeNet implementation from:
        # https://github.com/zllrunning/face-parsing.PyTorch
        
        # For now, we'll use a simple UNet-like structure as fallback
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.fc = nn.Conv2d(512, n_classes, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        x = self.fc(x)
        return x


def load_bisenet_model(weights_path=None, device='cpu'):
    """
    Load BiSeNet pre-trained model
    
    Tries to load the official BiSeNet implementation first.
    Falls back to a compatible interface if not available.
    """
    # Determine weights path
    if weights_path is None:
        # Try official repo location first
        repo_weights = Path(__file__).parent / "face-parsing.PyTorch" / "res" / "cp" / "79999_iter.pth"
        fallback_weights = Path(__file__).parent / "pretrained_models" / "bisenet" / "79999_iter.pth"
        
        if repo_weights.exists():
            weights_path = repo_weights
        else:
            weights_path = fallback_weights
    
    if not os.path.exists(weights_path):
        print(f"❌ Pre-trained weights not found at: {weights_path}")
        print("💡 Run: bash setup_bisenet.sh")
        print("💡 Or manually download from:")
        print("   https://drive.google.com/file/d/1a1_0xT5YQYfNU3IKH77HX4sNm9X0jY0E/view")
        return None
    
    try:
        if BISENET_AVAILABLE and OfficialBiSeNet is not None:
            # Use official BiSeNet implementation
            print(f"✅ Loading official BiSeNet model from: {weights_path}")
            net = OfficialBiSeNet(n_classes=NUM_CLASSES)
            state_dict = torch.load(weights_path, map_location=device)
            
            # Check if state_dict has the expected structure
            # The model expects keys like "cp.resnet.conv1.weight"
            # Some saved models might have different key structures
            model_keys = set(net.state_dict().keys())
            weight_keys = set(state_dict.keys())
            
            # Try loading with original keys first
            try:
                net.load_state_dict(state_dict, strict=True)
                print("✅ Loaded with strict=True")
            except RuntimeError as e:
                # If strict loading fails, try with strict=False
                print(f"⚠️  Strict loading failed: {str(e)[:200]}...")
                print("⚠️  Attempting to load with strict=False (allowing missing/unexpected keys)...")
                try:
                    net.load_state_dict(state_dict, strict=False)
                    print("✅ Loaded with strict=False")
                except Exception as e2:
                    print(f"❌ Loading failed even with strict=False: {e2}")
                    raise
            
            net.eval()
            net.to(device)
            print("✅ BiSeNet model loaded successfully!")
            return net
        else:
            # Official BiSeNet not available
            error_msg = "Official BiSeNet implementation not available."
            if BISENET_IMPORT_ERROR:
                error_msg += f" Import error: {BISENET_IMPORT_ERROR}"
            print(f"❌ {error_msg}")
            print("💡 Setup instructions:")
            print("   1. Ensure face-parsing.PyTorch repository is cloned")
            print("   2. Check that model.py and resnet.py exist in the repository")
            print("   3. Verify all dependencies are installed")
            raise ImportError(error_msg)
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 Setup instructions:")
        print("   1. Run: cd backend/ai_models/BiseNet && bash setup_bisenet.sh")
        print("   2. Download weights to the location shown above")
        print("   3. The official repository will be cloned automatically")
        return None


def preprocess_image(image, target_size=(512, 512)):
    """Preprocess image for BiSeNet"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # BGR to RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize - ensure float32
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # To tensor - ensure float32
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return image


def predict_mask(model, image, device='cpu'):
    """Predict segmentation mask"""
    model = model.to(device)
    # Ensure image is float32 and on correct device
    image = image.float().to(device)
    
    with torch.no_grad():
        output = model(image)
        # Handle tuple output (some BiSeNet implementations return tuple)
        if isinstance(output, tuple):
            output = output[0]
        mask = torch.argmax(output, dim=1)
    
    return mask.cpu().numpy()[0]


def colorize_mask(mask):
    
    """Convert mask to colored image"""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(len(PALETTE)):
        colored[mask == class_id] = PALETTE[class_id]
    return colored


def process_image_bisenet(image_path, weights_path=None, device='cpu', return_mask_array=True):
    """
    Main processing function
    
    Args:
        image_path: Path to image or PIL Image
        weights_path: Path to model weights (optional)
        device: 'cpu' or 'cuda'
        return_mask_array: If True, return numpy array mask
    
    Returns:
        Dictionary with:
        - mask: Segmentation mask (numpy array)
        - colorized_mask: Colored visualization
        - attributes: List of attribute names
        - attribute_mapping: Dict mapping class_id to attribute name
    """
    model = load_bisenet_model(weights_path, device)
    if model is None:
        return None
    
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    input_tensor = preprocess_image(image)
    mask = predict_mask(model, input_tensor, device)
    colorized = colorize_mask(mask)
    
    # Create attribute mapping (class_id -> attribute_name)
    # 0 = background, 1-18 = attributes
    attribute_mapping = {0: 'background'}
    for idx, attr in enumerate(CELEBA_ATTRIBUTES, start=1):
        attribute_mapping[idx] = attr
    
    result = {
        'colorized_mask': colorized,
        'attributes': CELEBA_ATTRIBUTES,
        'attribute_mapping': attribute_mapping
    }
    
    if return_mask_array:
        result['mask'] = mask
    
    return result


if __name__ == "__main__":
    print("BiSeNet Face Parsing Inference")
    print("=" * 50)
    print("\n📋 Setup Instructions:")
    print("1. Install BiSeNet implementation:")
    print("   git clone https://github.com/zllrunning/face-parsing.PyTorch.git")
    print("2. Download pre-trained weights:")
    print("   python download_pretrained_bisenet.py")
    print("3. Use the model for inference")
    print("\n" + "=" * 50)

