#!/usr/bin/env python3
"""
Test script for CelebAMask-HQ dataset loading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_models.unet.train_celeba_unet import CelebAMaskHQDataset, CELEBA_ATTRIBUTES
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def test_dataset_loading():
    """Test the CelebAMask-HQ dataset loading"""
    
    # Resolve data paths relative to backend directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(this_dir, "./"))
    img_dir = os.path.join(backend_dir, "data/CelebAMask-HQ/CelebA-HQ-img")
    mask_dir = os.path.join(backend_dir, "data/CelebAMask-HQ/CelebAMask-HQ-mask-anno")
    
    print(f"Image directory: {img_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Image directory exists: {os.path.exists(img_dir)}")
    print(f"Mask directory exists: {os.path.exists(mask_dir)}")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = CelebAMaskHQDataset(img_dir, mask_dir, transform=transform)
        print(f"Dataset created successfully!")
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading a few samples
        for i in range(min(3, len(dataset))):
            image, mask = dataset[i]
            print(f"Sample {i}:")
            print(f"  Image shape: {image.shape}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask unique values: {torch.unique(mask).tolist()}")
            print(f"  Mask value range: {mask.min().item()} to {mask.max().item()}")
            
            # Visualize the first sample
            if i == 0:
                # Denormalize image for visualization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_vis = image * std + mean
                img_vis = torch.clamp(img_vis, 0, 1)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(img_vis.permute(1, 2, 0))
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Mask
                axes[1].imshow(mask, cmap='tab20')
                axes[1].set_title('Segmentation Mask')
                axes[1].axis('off')
                
                # Overlay
                overlay = img_vis.permute(1, 2, 0).numpy()
                mask_vis = mask.numpy()
                
                # Create colored mask
                colored_mask = np.zeros((mask_vis.shape[0], mask_vis.shape[1], 3))
                for class_id in range(len(CELEBA_ATTRIBUTES) + 1):
                    colored_mask[mask_vis == class_id] = plt.cm.tab20(class_id)[:3]
                
                # Blend
                overlay = overlay * 0.7 + colored_mask * 0.3
                overlay = np.clip(overlay, 0, 1)
                
                axes[2].imshow(overlay)
                axes[2].set_title('Overlay')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig('celeba_test_sample.png', dpi=150, bbox_inches='tight')
                plt.show()
                
                print(f"  Visualization saved as 'celeba_test_sample.png'")
            
            print()
        
        print("Dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import torch
    test_dataset_loading() 