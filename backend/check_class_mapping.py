#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ai_models.BiseNet.inference_bisenet import CELEBA_ATTRIBUTES, PALETTE

print("=" * 60)
print("CELEBA_ATTRIBUTES Mapping:")
print("=" * 60)
for i, attr in enumerate(CELEBA_ATTRIBUTES, start=1):
    print(f"  class_id={i:2d} -> {attr:12s}")

print("\n" + "=" * 60)
print("Required Class IDs:")
print("=" * 60)
required = {
    2: 'nose',
    6: 'l_brow',
    7: 'r_brow',
    11: 'u_lip',
    12: 'l_lip'
}

for class_id, attr_name in required.items():
    if class_id <= len(CELEBA_ATTRIBUTES):
        actual_attr = CELEBA_ATTRIBUTES[class_id - 1]
        match = "✅" if actual_attr == attr_name else "❌"
        print(f"  {match} class_id={class_id:2d}: expected={attr_name:8s}, actual={actual_attr:8s}")
    else:
        print(f"  ❌ class_id={class_id:2d}: OUT OF RANGE (max={len(CELEBA_ATTRIBUTES)})")

