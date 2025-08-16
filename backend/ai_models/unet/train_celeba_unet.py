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
from sklearn.model_selection import train_test_split

# Define the 19 CelebAMask-HQ attributes
CELEBA_ATTRIBUTES = [
    'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 
    'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 
    'ear_r', 'neck', 'neck_l', 'cloth'
]

class CelebAMaskHQDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_size=(512, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get all image files
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        
        # Create image ID to mask files mapping
        self.img_to_masks = {}
        for img_file in self.img_files:
            img_id = os.path.splitext(os.path.basename(img_file))[0]
            mask_files = self._get_mask_files_for_image(img_id)
            if mask_files:  # Only include images that have masks
                self.img_to_masks[img_id] = mask_files
        
        # Filter to only include images with masks
        self.img_files = [f for f in self.img_files 
                         if os.path.splitext(os.path.basename(f))[0] in self.img_to_masks]
        
        print(f"Found {len(self.img_files)} images with masks")
    
    def _get_mask_files_for_image(self, img_id):
        """Get all mask files for a given image ID"""
        mask_files = []
        # CelebAMask-HQ mask files use zero-padded 5-digit IDs, e.g. 00000_hair.png
        padded_id = str(img_id).zfill(5)
        for subdir in os.listdir(self.mask_dir):
            subdir_path = os.path.join(self.mask_dir, subdir)
            if os.path.isdir(subdir_path):
                # Try zero-padded first (official format), fall back to non-padded just in case
                pattern = os.path.join(subdir_path, f"{padded_id}_*.png")
                files = glob.glob(pattern)
                if not files:
                    alt_pattern = os.path.join(subdir_path, f"{img_id}_*.png")
                    files = glob.glob(alt_pattern)
                mask_files.extend(files)
        return mask_files
    
    def _create_combined_mask(self, mask_files):
        """Create a combined mask from individual attribute masks"""
        # Initialize mask with zeros (background)
        combined_mask = np.zeros(self.target_size, dtype=np.uint8)
        
        for mask_file in mask_files:
            # Extract attribute name from filename
            filename = os.path.basename(mask_file)
            attr_name = filename.split('_', 1)[1].replace('.png', '')
            
            # Map attribute name to class index
            if attr_name in CELEBA_ATTRIBUTES:
                class_idx = CELEBA_ATTRIBUTES.index(attr_name) + 1  # +1 because 0 is background
                
                # Load and resize mask
                mask = Image.open(mask_file).convert('L')
                mask = mask.resize(self.target_size, Image.NEAREST)
                mask_array = np.array(mask)
                
                # Apply threshold to get binary mask
                binary_mask = (mask_array > 128).astype(np.uint8)
                
                # Add to combined mask
                combined_mask[binary_mask == 1] = class_idx
        
        return combined_mask
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_id = os.path.splitext(os.path.basename(img_file))[0]
        mask_files = self.img_to_masks[img_id]
        
        # Load and preprocess image
        image = Image.open(img_file).convert('RGB')
        image = image.resize(self.target_size, Image.BILINEAR)
        
        # Create combined mask
        mask = self._create_combined_mask(mask_files)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()
        
        return image, mask

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=len(CELEBA_ATTRIBUTES) + 1):  # +1 for background
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_layer(in_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        # Bottleneck
        self.bottleneck = self._make_layer(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._make_layer(1024, 512)  # 1024 = 512 + 512 (skip connection)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._make_layer(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._make_layer(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._make_layer(128, 64)
        
        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
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

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_celeba_unet.pth')
            print(f'  Saved best model (Val Loss: {val_loss:.4f})')
        
        print('-' * 50)
    
    return train_losses, val_losses

def visualize_predictions(model, val_loader, device, num_samples=5):
    """Visualize model predictions"""
    model.eval()
    
    # Get a batch from validation set
    images, masks = next(iter(val_loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Original image
        img = images[i].cpu().permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        gt_mask = masks[i].numpy()
        axes[i, 1].imshow(gt_mask, cmap='tab20')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        # Predicted mask
        pred_mask = predictions[i].cpu().numpy()
        axes[i, 2].imshow(pred_mask, cmap='tab20')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('celeba_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Resolve data paths relative to this file to be robust to working directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(this_dir, "../../"))
    img_dir = os.path.join(backend_dir, "data/CelebAMask-HQ/CelebA-HQ-img")
    mask_dir = os.path.join(backend_dir, "data/CelebAMask-HQ/CelebAMask-HQ-mask-anno")
    
    # Fallback: if the above doesn't exist, try relative to repo root
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        repo_root_fallback = os.path.abspath(os.path.join(backend_dir, "../"))
        alt_img_dir = os.path.join(repo_root_fallback, "backend/data/CelebAMask-HQ/CelebA-HQ-img")
        alt_mask_dir = os.path.join(repo_root_fallback, "backend/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno")
        if os.path.isdir(alt_img_dir) and os.path.isdir(alt_mask_dir):
            img_dir, mask_dir = alt_img_dir, alt_mask_dir
    
    print(f"Image directory: {img_dir} (exists: {os.path.isdir(img_dir)})")
    print(f"Mask directory: {mask_dir} (exists: {os.path.isdir(mask_dir)})")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CelebAMaskHQDataset(img_dir, mask_dir, transform=transform)
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Number of classes: {len(CELEBA_ATTRIBUTES) + 1}")  # +1 for background
    
    # Create model
    model = UNet(in_channels=3, out_channels=len(CELEBA_ATTRIBUTES) + 1)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=50, device=device)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('celeba_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Visualize predictions
    visualize_predictions(model, val_loader, device)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 