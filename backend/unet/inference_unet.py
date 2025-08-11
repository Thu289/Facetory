import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
import os
from ai_models.unet.train_unet import UNet

# --------- Inference Utilities ---------
NUM_CLASSES = 7  # Should match training
IMG_SIZE = 256
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints/best_unet.pth')

# Color palette for visualization (background, skin, lips, eyes, eyebrows, cheeks, other)
PALETTE = [
    (0, 0, 0),        # background
    (255, 224, 189),  # skin
    (255, 0, 0),      # lips
    (0, 255, 0),      # eyes
    (0, 0, 255),      # eyebrows
    (255, 255, 0),    # cheeks
    (128, 128, 128),  # other
]

def load_model(device='cpu'):
    model = UNet(n_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    return model

def preprocess_image(pil_img):
    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])
    return tf(pil_img).unsqueeze(0)  # (1, 3, H, W)

def predict_mask(model, pil_img, device='cpu'):
    img_tensor = preprocess_image(pil_img).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (H, W)
    return mask

def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        color_mask[mask == i] = color
    return Image.fromarray(color_mask)

# Example usage:
if __name__ == '__main__':
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device)
    img = Image.open(sys.argv[1]).convert('RGB')
    mask = predict_mask(model, img, device)
    color_mask = colorize_mask(mask)
    color_mask.save('predicted_mask.png')
    print('Saved predicted_mask.png') 