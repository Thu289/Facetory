# CelebAMask-HQ U-Net Training

This directory contains the U-Net model training and inference code for the CelebAMask-HQ dataset, which provides 19 different facial attributes for segmentation.

## Dataset Structure

The CelebAMask-HQ dataset should be organized as follows:

```
backend/data/CelebAMask-HQ/
├── CelebA-HQ-img/           # High-resolution face images (.jpg)
│   ├── 29999.jpg
│   ├── 29998.jpg
│   └── ...
├── CelebAMask-HQ-mask-anno/ # Mask annotations (.png)
│   ├── 0/                   # Subdirectories 0-14
│   │   ├── 00307_skin.png
│   │   ├── 00307_nose.png
│   │   ├── 00307_l_eye.png
│   │   └── ...
│   ├── 1/
│   └── ...
└── README.txt
```

## Facial Attributes

The model segments 19 facial attributes:

1. **skin** - Facial skin
2. **nose** - Nose
3. **eye_g** - Eye glasses
4. **l_eye** - Left eye
5. **r_eye** - Right eye
6. **l_brow** - Left eyebrow
7. **r_brow** - Right eyebrow
8. **l_ear** - Left ear
9. **r_ear** - Right ear
10. **mouth** - Mouth
11. **u_lip** - Upper lip
12. **l_lip** - Lower lip
13. **hair** - Hair
14. **hat** - Hat
15. **ear_r** - Ear rings
16. **neck** - Neck
17. **neck_l** - Necklace
18. **cloth** - Clothing

## Files

### Training
- `train_celeba_unet.py` - Main training script for CelebAMask-HQ dataset
- `inference_celeba_unet.py` - Inference script for trained model
- `test_celeba_dataset.py` - Test script to verify dataset loading

### Model Architecture
- U-Net with encoder-decoder structure
- 19 output channels (one for each attribute + background)
- Skip connections for better feature preservation
- Batch normalization and ReLU activations

## Usage

### 1. Test Dataset Loading

First, verify that your dataset is properly set up:

```bash
cd backend
python test_celeba_dataset.py
```

This will:
- Check if dataset directories exist
- Load a few samples
- Create a visualization of the first sample
- Save the visualization as `celeba_test_sample.png`

### 2. Train the Model

```bash
cd backend
python ai_models/unet/train_celeba_unet.py
```

Training parameters:
- **Epochs**: 50 (configurable)
- **Batch size**: 8
- **Learning rate**: 1e-4 with ReduceLROnPlateau scheduler
- **Image size**: 512x512
- **Train/Val split**: 80/20

The training will:
- Save the best model as `best_celeba_unet.pth`
- Plot training curves as `celeba_training_curves.png`
- Create prediction visualizations as `celeba_predictions.png`

### 3. Test Inference

```bash
cd backend
python ai_models/unet/inference_celeba_unet.py
```

This will test the trained model on a sample image.

### 4. Use in Backend API

The model is integrated into the FastAPI backend:

```python
# Endpoint: POST /api/makeup/celeba_unet_extract
# Returns:
{
    "colorized_mask": "data:image/png;base64,...",
    "annotated_image": "data:image/png;base64,...",
    "region_colors": {
        "skin": {"rgb": [120, 100, 80], "hex": "#786450"},
        "l_eye": {"rgb": [50, 30, 20], "hex": "#321e14"},
        # ... other attributes
    },
    "attributes": ["skin", "nose", "eye_g", ...]
}
```

## Model Output

The trained model provides:

1. **Segmentation Mask**: Pixel-level classification for each facial region
2. **Colorized Mask**: Visual representation with different colors for each attribute
3. **Annotated Image**: Original image overlaid with segmentation
4. **Region Colors**: Average RGB color for each detected facial region

## Performance

Expected performance metrics:
- **Training time**: ~2-4 hours on GPU (depending on dataset size)
- **Inference time**: ~0.1-0.5 seconds per image
- **Memory usage**: ~2-4 GB GPU memory during training

## Troubleshooting

### Common Issues

1. **"No such file or directory" errors**
   - Ensure CelebAMask-HQ dataset is properly downloaded and extracted
   - Check file paths in training script

2. **"CUDA out of memory"**
   - Reduce batch size in training script
   - Use smaller image size (e.g., 256x256)

3. **"Import torch could not be resolved"**
   - Install PyTorch: `pip install torch torchvision`
   - Ensure correct Python environment is activated

4. **Black segmentation masks**
   - Check if model is trained properly
   - Verify mask parsing logic in dataset loader
   - Ensure training data has proper annotations

### Debugging

1. **Test dataset loading first**:
   ```bash
   python test_celeba_dataset.py
   ```

2. **Check model predictions**:
   ```bash
   python ai_models/unet/inference_celeba_unet.py
   ```

3. **Monitor training progress**:
   - Check loss values in console output
   - Look at generated visualization files

## Integration with Frontend

The frontend can call the CelebAMask-HQ U-Net endpoint:

```typescript
const response = await fetch('/api/makeup/celeba_unet_extract', {
  method: 'POST',
  body: formData
});

const result = await response.json();
// result.colorized_mask - Base64 encoded colored mask
// result.annotated_image - Base64 encoded annotated image
// result.region_colors - Object with color information for each attribute
// result.attributes - Array of attribute names
```

## Next Steps

1. **Train the model** with your CelebAMask-HQ dataset
2. **Test inference** on sample images
3. **Integrate with frontend** for real-time makeup extraction
4. **Fine-tune** model parameters if needed
5. **Add data augmentation** for better generalization 