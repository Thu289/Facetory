# BiSeNet Face Parsing Integration

## Overview

BiSeNet (Bilateral Segmentation Network) for face parsing, trained on CelebAMask-HQ dataset with 19 facial attributes. This implementation follows the technology stack flow in `makeup_filter_system.md`.

## Setup

### 1. Install BiSeNet Repository

```bash
cd backend/ai_models/BiseNet
bash setup_bisenet.sh
```

This will:
- Clone the official BiSeNet repository from: https://github.com/zllrunning/face-parsing.PyTorch
- Set up the directory structure

### 2. Download Pre-trained Weights

Download the pre-trained weights (trained on CelebAMask-HQ):

**Option 1: Automatic (if available)**
```bash
python download_pretrained_bisenet.py
```

**Option 2: Manual Download**
1. Download from: https://drive.google.com/file/d/1a1_0xT5YQYfNU3IKH77HX4sNm9X0jY0E/view
2. Save to: `backend/ai_models/BiseNet/face-parsing.PyTorch/res/cp/79999_iter.pth`

Or save to fallback location:
`backend/ai_models/BiseNet/pretrained_models/bisenet/79999_iter.pth`

## Usage

### In Python Code

```python
from ai_models.BiseNet.inference_bisenet import process_image_bisenet

# Process an image
result = process_image_bisenet('path/to/image.jpg', device='cpu')

# Access results
mask = result['mask']  # Segmentation mask (numpy array)
colorized = result['colorized_mask']  # Colored visualization
attributes = result['attributes']  # List of 19 attributes
attribute_mapping = result['attribute_mapping']  # Dict: class_id -> attribute_name
```

### API Endpoint

The BiSeNet is integrated into the API via `/makeup/style_extract` endpoint:

```bash
curl -X POST "http://localhost:8000/makeup/style_extract" \
  -F "file=@image.jpg"
```

**Response includes:**
- Style parameters (lips, eyes, eyebrows, skin)
- Segmentation mask and annotations
- LAB color space analysis
- Coverage intensity, blend softness, texture type

## System Flow

Following `makeup_filter_system.md`:

```
1. Image Upload
   ↓
2. RetinaFace → Detect face region
   ↓
3. Crop face
   ↓
4. BiSeNet → Segment facial regions (19 attributes)
   ↓
5. Style Extraction → LAB color space, K-means, histogram
   ↓
6. Generate style parameters (JSON)
```

## CelebAMask-HQ Attributes

The model segments 19 facial regions:

1. skin
2. nose
3. eye_g (eye glasses)
4. l_eye (left eye)
5. r_eye (right eye)
6. l_brow (left eyebrow)
7. r_brow (right eyebrow)
8. l_ear (left ear)
9. r_ear (right ear)
10. mouth
11. u_lip (upper lip)
12. l_lip (lower lip)
13. hair
14. hat
15. ear_r (earring)
16. neck
17. neck_l (necklace)
18. cloth

## Dependencies

All required dependencies should already be in `backend/requirements.txt`:
- torch >= 2.0.0
- torchvision
- opencv-python
- numpy
- pillow
- scikit-learn (for style extraction K-means)

## Troubleshooting

### Error: "Pre-trained weights not found"
- Ensure you've downloaded the weights to the correct location
- Check the path in error message and download accordingly

### Error: "Official BiSeNet implementation required"
- Run `bash setup_bisenet.sh` to clone the official repository
- The code will automatically use the official implementation if available

### Model Performance
- On GPU: ~50 FPS
- On CPU: ~2-5 FPS (slower but works)

## Integration with Style Extraction

The BiSeNet output is automatically processed by `app/services/style_extraction.py` which:
- Converts RGB to LAB color space
- Applies K-means clustering (k=3) per region
- Analyzes histogram distributions
- Extracts style parameters (coverage, blend softness, texture type)

See `app/api/face_detection.py` → `/makeup/style_extract` endpoint for complete integration.

