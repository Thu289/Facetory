# Pre-trained Models cho Face Parsing

## Tổng quan

Thay vì train từ đầu, bạn có thể sử dụng các pre-trained models đã được train trên CelebAMask-HQ dataset với 19 facial attributes.

## 🏆 Recommended Models

### 1. BiSeNet Face Parsing (Khuyến nghị)

**Ưu điểm:**
- ✅ Model nhẹ, nhanh (~30-50 FPS trên GPU)
- ✅ Chạy tốt trên CPU (chậm hơn nhưng vẫn được)
- ✅ Pre-trained trên CelebAMask-HQ (19 classes)
- ✅ Code và weights có sẵn công khai

**Cài đặt:**

```bash
# Clone repository
git clone https://github.com/zllrunning/face-parsing.PyTorch.git
cd face-parsing.PyTorch

# Download pre-trained weights
# Weights sẽ được tải tự động hoặc download từ link trong repo
# Thường là file: 79999_iter.pth (cho CelebAMask-HQ)
```

**Sử dụng:**

```python
from inference_bisenet import process_image_bisenet

result = process_image_bisenet('path/to/image.jpg')
mask = result['mask']
colorized = result['colorized_mask']
```

**Links:**
- GitHub: https://github.com/zllrunning/face-parsing.PyTorch
- Pre-trained weights: Thường có trong releases hoặc Google Drive link trong README

---

### 2. Face-Parsing-Net (FAN-based)

**Repository:** 
- https://github.com/YudongGuo/AD-NeRF (có face parsing module)

---

### 3. MediaPipe Face Segmentation

**Ưu điểm:**
- ✅ Rất nhẹ và nhanh
- ✅ Chạy tốt trên CPU
- ✅ Không cần GPU

**Nhược điểm:**
- ❌ Chỉ có 2-3 classes (face/skin/hair)
- ❌ Không phù hợp nếu cần 19 attributes chi tiết

**Sử dụng:**

```python
import mediapipe as mp

mp_face_segmentation = mp.solutions.face_segmentation
face_segmentation = mp_face_segmentation.FaceSegmentation(model_selection=1)

# Đã được tích hợp sẵn trong codebase của bạn
```

---

## 📥 Quick Start với BiSeNet

### Option 1: Sử dụng trực tiếp từ repository gốc

```bash
cd /tmp
git clone https://github.com/zllrunning/face-parsing.PyTorch.git
cd face-parsing.PyTorch

# Download weights (thường có link trong README hoặc releases)
# Sau đó sử dụng script của họ để inference
```

### Option 2: Tích hợp vào codebase hiện tại

1. **Download weights:**
```bash
cd backend/ai_models/unet
python download_pretrained_bisenet.py
```

2. **Copy BiSeNet model code:**
   - Clone repository gốc
   - Copy các files cần thiết (`model.py`, `resnet.py`, etc.)
   - Hoặc install như package

3. **Sử dụng trong API:**
   - Import `inference_bisenet` thay vì `inference_celeba_unet`
   - Hoặc tạo wrapper để switch giữa models

---

## 🔄 Migration Guide

### Thay thế UNet bằng BiSeNet

**File:** `backend/app/api/face_detection.py`

```python
# Thay đổi import
# from ai_models.unet.inference_celeba_unet import process_image_with_celeba_unet
from ai_models.unet.inference_bisenet import process_image_bisenet

# Sử dụng trong endpoint
result = process_image_bisenet(image_path, device=device)
```

---

## 📊 So sánh Models

| Model | Speed (GPU) | Speed (CPU) | Accuracy | Size | Classes |
|-------|-------------|-------------|----------|------|---------|
| BiSeNet | ~50 FPS | ~2-5 FPS | ⭐⭐⭐⭐ | ~40MB | 19 |
| UNet (custom) | ~20 FPS | ~0.5-1 FPS | ⭐⭐⭐⭐ | ~60MB | 19 |
| MediaPipe | N/A | ~30 FPS | ⭐⭐ | ~10MB | 2-3 |

---

## 💡 Recommendations

1. **Nếu cần tốc độ cao + CPU:** MediaPipe (nhưng chỉ 2-3 classes)
2. **Nếu cần 19 classes + có GPU:** BiSeNet
3. **Nếu cần 19 classes + chỉ CPU:** BiSeNet vẫn tốt hơn train từ đầu
4. **Nếu muốn customize:** Train từ đầu với dataset riêng

---

## 🔗 Useful Links

- BiSeNet Paper: https://arxiv.org/abs/1808.00897
- CelebAMask-HQ Dataset: https://github.com/switchablenorms/CelebAMask-HQ
- Face Parsing Survey: https://github.com/JDAI-CV/FaceX-Zoo

---

## ⚠️ Notes

- Pre-trained models thường được train trên CelebAMask-HQ (30,000 ảnh)
- Có thể cần fine-tune trên dataset riêng nếu domain khác
- Kiểm tra license của pre-trained weights trước khi dùng commercial

