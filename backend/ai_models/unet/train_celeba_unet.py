import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

# Define the 19 CelebAMask-HQ attributes
CELEBA_ATTRIBUTES = [
    'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 
    'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 
    'ear_r', 'neck', 'neck_l', 'cloth'
]

class CelebAMaskHQDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_indices=None, transform=None, target_size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.image_indices = image_indices
        
        self.total_images = len(self.image_indices)
        print(f"Dataset: {self.total_images} images (IDs: {min(self.image_indices)}-{max(self.image_indices)})")
    
    def _get_image_path(self, img_id):
        """Get image path for given ID"""
        return os.path.join(self.img_dir, f"{img_id}.jpg")
    
    def _get_mask_files_for_image(self, img_id):
        """Get all mask files for a given image ID"""
        mask_files = []
        padded_id = str(img_id).zfill(5)
        for subdir in os.listdir(self.mask_dir):
            subdir_path = os.path.join(self.mask_dir, subdir)
            if os.path.isdir(subdir_path):
                pattern = os.path.join(subdir_path, f"{padded_id}_*.png")
                files = glob.glob(pattern)
                if not files:
                    alt_pattern = os.path.join(subdir_path, f"{img_id}_*.png")
                    files = glob.glob(alt_pattern)
                mask_files.extend(files)
        return mask_files
    
    def _create_combined_mask(self, mask_files):
        """Create a combined mask from individual attribute masks"""
        combined_mask = np.zeros(self.target_size, dtype=np.uint8)
        
        for mask_file in mask_files:
            filename = os.path.basename(mask_file)
            attr_name = filename.split('_', 1)[1].replace('.png', '')
            
            if attr_name in CELEBA_ATTRIBUTES:
                class_idx = CELEBA_ATTRIBUTES.index(attr_name) + 1
                
                mask = Image.open(mask_file).convert('L')
                mask = mask.resize(self.target_size, Image.NEAREST)
                mask_array = np.array(mask)
                
                binary_mask = (mask_array > 128).astype(np.uint8)
                combined_mask[binary_mask == 1] = class_idx
        
        return combined_mask
    
    def __len__(self):
        return self.total_images
    
    def __getitem__(self, idx):
        img_id = self.image_indices[idx]
        img_path = self._get_image_path(img_id)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_id}.jpg not found")
        
        mask_files = self._get_mask_files_for_image(img_id)
        if not mask_files:
            raise ValueError(f"No masks found for image {img_id}")
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.target_size, Image.BILINEAR)
        
        mask = self._create_combined_mask(mask_files)
        
        if self.transform:
            image = self.transform(image)
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=len(CELEBA_ATTRIBUTES) + 1):
        super(UNet, self).__init__()
        
        # Encoder (reduced channels)
        self.enc1 = self._make_layer(in_channels, 32)
        self.enc2 = self._make_layer(32, 64)
        self.enc3 = self._make_layer(64, 128)
        self.enc4 = self._make_layer(128, 256)
        
        # Bottleneck
        self.bottleneck = self._make_layer(256, 512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._make_layer(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._make_layer(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._make_layer(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._make_layer(64, 32)
        
        # Final output layer
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        
        # Decoder with skip connections
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

def train_model(model, train_loader, test_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    model_save_path = 'checkpoints/best_unet.pth'
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (images, masks) in enumerate(train_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{train_loss/(batch_idx+1):.4f}'
            })
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = 0
        test_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_bar):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                test_loss += loss.item()
                test_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{test_loss/(batch_idx+1):.4f}'
                })
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Test Loss: {test_loss:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'  ✅ Saved best model (Test Loss: {test_loss:.4f})')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, checkpoint_path)
            print(f'  💾 Saved checkpoint: {checkpoint_path}')
        
        print('-' * 50)
        
        # Clear memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    return train_losses, test_losses

def visualize_predictions(model, val_loader, device, num_samples=3):
    """Visualize model predictions"""
    model.eval()
    
    images, masks = next(iter(val_loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        img = images[i].cpu().permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        gt_mask = masks[i].numpy()
        axes[i, 1].imshow(gt_mask, cmap='tab20')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        pred_mask = predictions[i].cpu().numpy()
        axes[i, 2].imshow(pred_mask, cmap='tab20')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('checkpoints/celeba_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Visualization saved to checkpoints/celeba_predictions.png")

def create_train_test_split(total_images=30000, test_ratio=0.2, random_seed=42):
    """Create train/test split indices without loading images"""
    import random
    random.seed(random_seed)
    
    all_indices = list(range(total_images))
    random.shuffle(all_indices)
    
    test_size = int(total_images * test_ratio)
    test_indices = all_indices[:test_size]
    train_indices = all_indices[test_size:]
    
    return train_indices, test_indices

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_pin_memory = torch.cuda.is_available()
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Data paths
    img_dir = "data/CelebAMask-HQ/CelebA-HQ-img"
    mask_dir = "data/CelebAMask-HQ/CelebAMask-HQ-mask-anno"
    
    print(f"Image directory: {img_dir} (exists: {os.path.isdir(img_dir)})")
    print(f"Mask directory: {mask_dir} (exists: {os.path.isdir(mask_dir)})")
    
    # Create train/test split
    print("📊 Creating train/test split...")
    train_indices, test_indices = create_train_test_split(
        total_images=30000, 
        test_ratio=0.2, 
        random_seed=42
    )
    
    print(f"✅ Split created: {len(train_indices)} train, {len(test_indices)} test")
    
    # Create datasets (reduced image size to 256x256)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CelebAMaskHQDataset(
        img_dir, mask_dir, 
        image_indices=train_indices, 
        transform=transform,
        target_size=(256, 256)  # Reduced from 512
    )
    test_dataset = CelebAMaskHQDataset(
        img_dir, mask_dir, 
        image_indices=test_indices, 
        transform=transform,
        target_size=(256, 256)  # Reduced from 512
    )
    
    # Create data loaders with smaller batch size
    BATCH_SIZE = 8  # Reduced from 50
    NUM_WORKERS = 0 if not use_pin_memory else 2  # No workers on CPU
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=False  # Changed to False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=False  # Changed to False
    )
    
    print(f"🚀 Train samples: {len(train_dataset)}")
    print(f"🧪 Test samples: {len(test_dataset)}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"👥 Workers: {NUM_WORKERS}")
    print(f"Number of classes: {len(CELEBA_ATTRIBUTES) + 1}")
    
    # Create model
    model = UNet(in_channels=3, out_channels=len(CELEBA_ATTRIBUTES) + 1)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Check if model already exists
    model_path = 'checkpoints/best_unet.pth'
    if os.path.exists(model_path):
        print(f"🔄 Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model loaded successfully!")
        
        # Run evaluation
        print("\n📊 Running evaluation on test set...")
        model.eval()
        test_loss = 0
        test_bar = tqdm(test_loader, desc="Evaluation")
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_bar):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, masks)
                test_loss += loss.item()
                test_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        test_loss /= len(test_loader)
        print(f"\n📈 Final Test Loss: {test_loss:.4f}")
        
    else:
        print("🚀 Starting training...")
        train_losses, test_losses = train_model(
            model, train_loader, test_loader, 
            num_epochs=50, 
            device=device
        )
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.savefig('checkpoints/training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Training curves saved to checkpoints/training_curves.png")
        
        print("✅ Training completed!")
    
    # Visualize predictions
    print("\n🎨 Creating visualizations...")
    visualize_predictions(model, test_loader, device, num_samples=3)
    
    print(f"\n🎯 Model ready for inference!")
    print(f"📁 Best model saved at: {model_path}")

if __name__ == "__main__":
    main()