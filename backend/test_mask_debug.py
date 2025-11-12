#!/usr/bin/env python3
"""
Test script to debug mask creation issues
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from ai_models.BiseNet.inference_bisenet import process_image_bisenet, CELEBA_ATTRIBUTES
import torch

# Test with a sample image path
test_image = sys.argv[1] if len(sys.argv) > 1 else None
if not test_image or not os.path.exists(test_image):
    print("Usage: python test_mask_debug.py <image_path>")
    print("Example: python test_mask_debug.py /tmp/test.jpg")
    sys.exit(1)

print("=" * 80)
print("BiSeNet MASK DEBUG TEST")
print("=" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Process image
print(f"\n📸 Processing image: {test_image}")
result = process_image_bisenet(test_image, device=device, return_mask_array=True)

if result is None:
    print("❌ BiSeNet processing failed!")
    sys.exit(1)

mask = result['mask']
attribute_mapping = result['attribute_mapping']
colorized = result['colorized_mask']

print(f"\n✅ BiSeNet processing successful!")
print(f"   Mask shape: {mask.shape}")
print(f"   Mask dtype: {mask.dtype}")
print(f"   Mask min/max: {mask.min()}/{mask.max()}")

# Check detected classes
detected_classes = np.unique(mask)
print(f"\n🔍 Detected class IDs in mask: {detected_classes}")
print(f"   Total unique classes: {len(detected_classes)}")

# Check attribute mapping
print(f"\n🔍 Attribute mapping:")
for class_id in sorted(attribute_mapping.keys()):
    attr = attribute_mapping[class_id]
    pixels = np.sum(mask == class_id)
    print(f"   class_id={class_id:2d}: {attr:12s} - {pixels:6d} pixels")

# Check specific attributes we need
print(f"\n🔍 Checking required attributes:")
required = {
    11: 'u_lip',
    12: 'l_lip',
    6: 'l_brow',
    7: 'r_brow',
    2: 'nose'
}

for class_id, attr_name in required.items():
    in_mapping = class_id in attribute_mapping
    in_detected = class_id in detected_classes
    pixels = np.sum(mask == class_id) if in_detected else 0
    status = "✅" if pixels > 0 else "❌"
    print(f"   {status} class_id={class_id:2d} ({attr_name:8s}): "
          f"in_mapping={in_mapping:5s}, in_detected={in_detected:5s}, pixels={pixels:6d}")

# Test mask creation
print(f"\n🧪 Testing mask creation:")
h, w = mask.shape

# Test lips mask
print(f"   Creating lips mask...")
lips_mask = np.zeros((h, w), dtype=np.float32)
for class_id in [11, 12]:  # u_lip, l_lip
    if class_id in detected_classes:
        pixels = np.sum(mask == class_id)
        print(f"      Adding class_id={class_id} ({attribute_mapping.get(class_id, 'unknown')}): {pixels} pixels")
        lips_mask[mask == class_id] = 1.0

lips_pixels = np.sum(lips_mask > 0)
print(f"   ✅ Lips mask: {lips_pixels} pixels")

# Test eyebrows mask
print(f"   Creating eyebrows mask...")
brows_mask = np.zeros((h, w), dtype=np.float32)
for class_id in [6, 7]:  # l_brow, r_brow
    if class_id in detected_classes:
        pixels = np.sum(mask == class_id)
        print(f"      Adding class_id={class_id} ({attribute_mapping.get(class_id, 'unknown')}): {pixels} pixels")
        brows_mask[mask == class_id] = 1.0

brows_pixels = np.sum(brows_mask > 0)
print(f"   ✅ Eyebrows mask: {brows_pixels} pixels")

# Test nose mask
print(f"   Creating nose mask...")
nose_mask = np.zeros((h, w), dtype=np.float32)
if 2 in detected_classes:
    pixels = np.sum(mask == 2)
    print(f"      Adding class_id=2 ({attribute_mapping.get(2, 'unknown')}): {pixels} pixels")
    nose_mask[mask == 2] = 1.0

nose_pixels = np.sum(nose_mask > 0)
print(f"   ✅ Nose mask: {nose_pixels} pixels")

print("\n" + "=" * 80)

