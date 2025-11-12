"""
Script to download and setup BiSeNet pre-trained face parsing model
This model is trained on CelebAMask-HQ dataset with 19 facial attributes
"""

import os
import torch
import urllib.request
from pathlib import Path

# BiSeNet pre-trained weights URL (CelebAMask-HQ)
BISENET_WEIGHTS_URL = "https://github.com/zllrunning/face-parsing.PyTorch/raw/master/res/cp/79999_iter.pth"
BISENET_MODEL_DIR = Path(__file__).parent / "pretrained_models" / "bisenet"
BISENET_WEIGHTS_PATH = BISENET_MODEL_DIR / "79999_iter.pth"

def download_bisenet_weights():
    """Download BiSeNet pre-trained weights"""
    os.makedirs(BISENET_MODEL_DIR, exist_ok=True)
    
    if os.path.exists(BISENET_WEIGHTS_PATH):
        print(f"✅ Pre-trained weights already exist: {BISENET_WEIGHTS_PATH}")
        return BISENET_WEIGHTS_PATH
    
    print(f"📥 Downloading BiSeNet pre-trained weights...")
    print(f"   URL: {BISENET_WEIGHTS_URL}")
    print(f"   Saving to: {BISENET_WEIGHTS_PATH}")
    
    try:
        urllib.request.urlretrieve(BISENET_WEIGHTS_URL, BISENET_WEIGHTS_PATH)
        print(f"✅ Download completed!")
        return BISENET_WEIGHTS_PATH
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n💡 Alternative: Manually download from:")
        print("   https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view")
        print(f"   And save to: {BISENET_WEIGHTS_PATH}")
        return None

if __name__ == "__main__":
    download_bisenet_weights()

