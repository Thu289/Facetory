# 📋 Flow Apply Filter Trên Image

## 🔄 Quy Trình Chi Tiết

### 1️⃣ **API Endpoint**
```
POST /api/makeup/style/apply_filter
```
**Input:**
- `file`: Image file (UploadFile)
- `style_id`: ID của style cần áp dụng
- `intensity`: Mức độ filter (0.0 - 1.0), mặc định = 1.0

---

### 2️⃣ **Load Style Data**
```python
style_data = get_style(style_id)  # Load từ /tmp/facetory_styles/{style_id}.json
```

**Style data structure:**
```json
{
  "style_id": "style_abc123",
  "download_urls": {
    "luts": {
      "lips": "/api/makeup/storage/file/styles/style_abc123/luts/style_abc123_lips_lut.bin",
      "eyes": "/api/makeup/storage/file/styles/style_abc123/luts/style_abc123_eyes_lut.bin",
      "eyebrows": "...",
      "skin": "..."
    },
    "shaders": {...},
    "style_parameters": "...",
    "thumbnail": "..."
  },
  "style_parameters": {
    "lips": {
      "primary_lab": [50, 20, 15],
      "secondary_lab": [45, 18, 12],
      "coverage_intensity": 0.8,
      "blend_softness": 0.6,
      ...
    },
    "eyes": {...},
    ...
  }
}
```

**Lưu ý:**
- ❌ Style data KHÔNG chứa LUT data (chỉ có URLs)
- ✅ LUTs đã được generate và lưu trong MinIO KHI TẠO STYLE
- ✅ Style parameters chỉ để tham khảo, không dùng khi apply

---

### 3️⃣ **Face Detection & Segmentation**

**Step 3.1: RetinaFace Detection**
```python
face_results = RetinaFace.detect_faces(image_path)
first_face = list(face_results.values())[0]
face_box = first_face["facial_area"]  # [x1, y1, x2, y2]
```

**Step 3.2: Crop Face**
```python
padding = int((x2 - x1) * 0.2)  # 20% padding
x1_crop = max(0, x1 - padding)
y1_crop = max(0, y1 - padding)
x2_crop = min(w, x2 + padding)
y2_crop = min(h, y2 + padding)
cropped_face = image[y1_crop:y2_crop, x1_crop:x2_crop]
```

**Step 3.3: BiSeNet Segmentation**
```python
bisenet_result = process_image_bisenet(cropped_path, device='cpu', return_mask_array=True)
segmentation_mask = bisenet_result['mask']  # Shape: (512, 512) hoặc (H, W)
attribute_mapping = bisenet_result['attribute_mapping']  # {class_id: 'u_lip', ...}
```

**Step 3.4: Create Region Masks**
```python
region_masks = {
    'lips': lips_mask_full,      # Chỉ u_lip + l_lip (exclude mouth, nose)
    'eyes': eyes_mask_full,       # Chỉ l_eye + r_eye (exclude eye_g, brows)
    'eyebrows': eyebrows_mask_full,  # Chỉ l_brow + r_brow
    'nose': nose_mask_full,       # Chỉ nose (exclude brows)
    'skin': skin_mask_full        # Skin regions
}

face_mask = (segmentation_mask > 0)  # Tất cả non-background pixels
```

---

### 4️⃣ **Apply Filter: `apply_style_to_image()`**

**Flow cho mỗi region (lips, eyes, eyebrows, skin):**

#### 4.1. Extract LUT URL
```python
lut_url = style_data['download_urls']['luts'][region]
# Ví dụ: "/api/makeup/storage/file/styles/style_abc123/luts/style_abc123_lips_lut.bin"
```

#### 4.2. Extract Object Name từ Proxy URL
```python
object_name = lut_url.replace('/api/makeup/storage/file/', '')
# → "styles/style_abc123/luts/style_abc123_lips_lut.bin"
object_name = unquote(object_name)  # Decode URL encoding
```

#### 4.3. Download LUT từ MinIO
```python
minio_service = MinioService()
temp_lut_path = tempfile.mktemp(suffix='.bin')

minio_service.client.fget_object(
    bucket_name="facetory-storage",
    object_name=object_name,
    file_path=temp_lut_path
)
```

#### 4.4. Load LUT Binary
```python
from app.services.lut_generation import load_lut_binary

lut = load_lut_binary(temp_lut_path)
# lut shape: (32, 32, 32, 3)  # 3D LUT với 32x32x32 grid
# lut dtype: uint8
```

**LUT Binary Format:**
- Header: 4 int32 values [R_size, G_size, B_size, channels] = [32, 32, 32, 3]
- Data: uint8 array, shape (32, 32, 32, 3)

#### 4.5. Apply LUT to Image
```python
filtered_region = apply_lut_efficient(image, lut, intensity)
```

**LUT Application Logic:**
```python
# 1. Normalize image to [0, 1]
img_normalized = image.astype(np.float32) / 255.0

# 2. Map RGB values to LUT indices
r_indices = (pixels[:, 0] * 31).astype(int)  # 0-31
g_indices = (pixels[:, 1] * 31).astype(int)
b_indices = (pixels[:, 2] * 31).astype(int)

# 3. Lookup colors from LUT
filtered_colors = lut[r_indices, g_indices, b_indices] / 255.0

# 4. Blend with original
blended = original_pixels * (1 - intensity) + filtered_colors * intensity

# 5. Convert back to uint8
filtered = (blended * 255).astype(np.uint8)
```

#### 4.6. Apply Region Mask
```python
if use_regions and region_masks and region in region_masks:
    region_mask = region_masks[region]
    
    # Resize mask if needed
    if region_mask.shape != image.shape[:2]:
        region_mask = cv2.resize(region_mask, (W, H), INTER_NEAREST)
    
    # Normalize to [0, 1]
    if region_mask.max() > 1.0:
        region_mask = region_mask / 255.0
    
    # Combine with face mask
    region_mask = region_mask * face_mask
    
    # Only apply if region exists
    if np.any(region_mask > 0):
        filtered_results.append({
            'filtered': filtered_region,
            'mask': region_mask,
            'weight': 1.0
        })
```

---

### 5️⃣ **Blend All Regions**

```python
result = image.astype(np.float32).copy()

for item in filtered_results:
    filtered = item['filtered']      # Filtered image for this region
    mask = item['mask']              # Region mask (2D)
    weight = item['weight']           # 1.0
    
    # Expand mask to 3D for RGB
    mask_3d = np.stack([mask, mask, mask], axis=-1)  # (H, W, 3)
    
    # Blend formula
    result = result * (1 - mask_3d * weight) + filtered * mask_3d * weight
```

**Blend Formula:**
```
result_pixel = original_pixel * (1 - mask_value) + filtered_pixel * mask_value
```

**Nếu mask = 1.0 (100% region):**
- `result = filtered` (full filter)

**Nếu mask = 0.5 (50% region):**
- `result = 0.5 * original + 0.5 * filtered` (50% blend)

**Nếu mask = 0.0 (outside region):**
- `result = original` (no filter)

---

### 6️⃣ **Final Output**

```python
# Clip to valid range [0, 255]
result = np.clip(result, 0, 255).astype(np.uint8)

# Save with high quality
filtered_img = Image.fromarray(result)
filtered_img.save(output_path, quality=98, optimize=False)

# Encode to base64
filtered_b64 = base64.b64encode(filtered_data).decode("utf-8")
```

**API Response:**
```json
{
  "success": true,
  "style_id": "style_abc123",
  "intensity": 1.0,
  "filtered_image": "data:image/jpeg;base64,...",
  "original_size": [1920, 1080],
  "mask_previews": {
    "lips": "data:image/png;base64,...",
    "eyes": "data:image/png;base64,...",
    ...
  },
  "regions_detected": ["lips", "eyes", "eyebrows", "nose", "skin"]
}
```

---

## 💡 Điểm Quan Trọng

### ✅ LUTs được generate TRƯỚC (khi tạo style):
- Style parameters (primary_lab, coverage_intensity, ...) → Generate 3D LUT
- LUT được lưu vào MinIO dưới dạng binary file (.bin)
- Style metadata chỉ lưu URL, không lưu LUT data

### ✅ Khi apply filter:
- **KHÔNG generate lại LUT** từ style parameters
- **Chỉ download và apply** LUT đã có sẵn
- Nhanh hơn và đảm bảo consistency

### ✅ Style Parameters:
- `primary_lab`, `secondary_lab`: Dùng để GENERATE LUT (khi tạo style)
- `coverage_intensity`, `blend_softness`: Dùng để tính blend factor trong LUT generation
- **KHÔNG dùng trực tiếp** khi apply filter

### ✅ Region Masks:
- Được tạo từ BiSeNet segmentation trên **target image** (image cần apply filter)
- **KHÔNG dùng masks từ style creation image**
- Mỗi image có masks riêng, đảm bảo filter apply đúng vị trí

---

## 🔍 Debug Flow

**Để debug, check:**
1. Style data có `download_urls.luts` không?
2. LUT URLs có valid không? (proxy URLs)
3. Download từ MinIO thành công không?
4. Region masks có detected đúng không?
5. LUT application có blend đúng không?

**Debug outputs:**
- `mask_previews`: Visualize region masks
- `regions_detected`: List regions found
- Console logs: Pixel counts, class IDs, warnings

