# API URLs - Facetory System

## 🌐 Base URLs

### Backend API
```
http://localhost:8000
```

### Frontend (Next.js)
```
http://localhost:3000
```

---

## 📍 Backend API Endpoints

### Base URL: `http://localhost:8000`

#### Health Check
```
GET http://localhost:8000/health
```

#### Root
```
GET http://localhost:8000/
```

---

### Face Detection APIs (`/api/face/...`)

#### 1. Detect Faces
```
POST http://localhost:8000/api/face/detect
Content-Type: multipart/form-data
Body: file (image file)
```

#### 2. Crop Face
```
POST http://localhost:8000/api/face/crop
Content-Type: multipart/form-data
Body: 
  - file (image file)
  - x1, y1, x2, y2 (bounding box coordinates)
```

#### 3. Extract Makeup (MediaPipe)
```
POST http://localhost:8000/api/face/makeup/extract
Content-Type: multipart/form-data
Body: file (image file)
```

#### 4. U-Net Extract
```
POST http://localhost:8000/api/face/makeup/unet_extract
Content-Type: multipart/form-data
Body: file (image file)
```

#### 5. CelebAMask-HQ U-Net Extract
```
POST http://localhost:8000/api/face/makeup/celeba_unet_extract
Content-Type: multipart/form-data
Body: file (image file)
```

#### 6. **Style Extract (BiSeNet + Style Extraction)** ⭐ NEW
```
POST http://localhost:8000/api/face/makeup/style_extract
Content-Type: multipart/form-data
Body: file (image file)
```

**Response includes:**
- `style_id`: Unique style identifier
- `style_parameters`: Lips, eyes, eyebrows, skin with LAB colors, coverage, blend softness
- `segmentation`: Colorized mask, annotated image
- `face_detection`: Bounding box info
- `processing_info`: Device and model info

---

### Upload APIs (`/api/upload/...`)

Check `backend/app/api/upload.py` for upload endpoints

---

### Auth APIs (`/api/auth/...`)

Check `backend/app/api/auth.py` for authentication endpoints

---

## 🖥️ Frontend URLs

### Base URL: `http://localhost:3000`

#### Home Page
```
GET http://localhost:3000/
```

#### API Proxy
Frontend proxies API calls through Next.js:
- Frontend calls: `/api/face/detect`
- Next.js rewrites to: `http://backend:8000/api/face/detect` (in Docker)
- Or: `http://localhost:8000/api/face/detect` (local development)

---

## 🧪 Testing với cURL

### Test Style Extract Endpoint (BiSeNet)

```bash
curl -X POST "http://localhost:8000/api/face/makeup/style_extract" \
  -F "file=@path/to/your/image.jpg" \
  -o response.json
```

### Test Face Detection

```bash
curl -X POST "http://localhost:8000/api/face/detect" \
  -F "file=@path/to/your/image.jpg"
```

---

## 🐳 Docker URLs

Nếu chạy với Docker Compose:

### Inside Docker Network
- Backend: `http://backend:8000` (service name)
- Frontend: `http://frontend:3000` (service name)

### From Host Machine
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

---

## 📋 API Documentation

### Swagger/OpenAPI Docs (nếu có)
```
http://localhost:8000/docs
```

### ReDoc (nếu có)
```
http://localhost:8000/redoc
```

---

## 🔗 Quick Reference

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | `http://localhost:3000` | Next.js web application |
| Backend API | `http://localhost:8000` | FastAPI backend |
| Health Check | `http://localhost:8000/health` | API health status |
| Style Extract | `http://localhost:8000/api/face/makeup/style_extract` | BiSeNet + Style Extraction ⭐ |

---

## 📝 Notes

- Đảm bảo backend đã chạy trước khi gọi API
- Frontend cần backend để hoạt động
- BiSeNet model weights phải được setup (xem `backend/ai_models/BiseNet/README.md`)
- CORS được cấu hình cho `http://localhost:3000`

