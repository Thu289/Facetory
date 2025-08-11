import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# --------- U-Net Model ---------
class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.center = CBR(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, n_classes, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        out = self.final(d1)
        return out

# --------- Dataset Loader ---------
class HelenFaceDataset(Dataset):
    def __init__(self, data_dir, img_size=256, n_classes=7):
        self.img_paths = sorted([p for p in glob.glob(os.path.join(data_dir, '*.jpg'))])
        self.label_paths = [p.replace('.jpg', '.xml') for p in self.img_paths]
        self.img_size = img_size
        self.n_classes = n_classes
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        # No mask_transform needed, as mask will be generated from XML

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.img_transform(img)
        # Parse XML to create mask
        xml_path = self.label_paths[idx]
        mask = self.parse_xml_to_mask(xml_path)
        mask = torch.from_numpy(mask).long()
        return img, mask

    def parse_xml_to_mask(self, xml_path):
        # Placeholder: parse the XML and create a mask of shape (img_size, img_size)
        # with integer values for each class (0 to n_classes-1)
        # You need to implement this based on your XML format
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        # TODO: Parse XML and fill mask with class indices
        return mask

# --------- Training Script ---------
def train():
    # Updated dataset paths
    base_dir = 'data/HelenDataset'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    n_classes = 7  # e.g. background, skin, lips, eyes, eyebrows, cheeks, other
    batch_size = 8
    lr = 1e-3
    n_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HelenFaceDataset(train_dir, img_size=256, n_classes=n_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = UNet(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    os.makedirs('backend/ai_models/unet/checkpoints', exist_ok=True)
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(loader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f'Epoch {epoch+1} Loss: {epoch_loss:.4f}')
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'backend/ai_models/unet/checkpoints/best_unet.pth')
            print('Saved best model!')

if __name__ == '__main__':
    train() 