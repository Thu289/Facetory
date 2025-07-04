# Facetory - AI Face Filter System

## 📋 Tổng quan dự án

Facetory là một hệ thống web AI cho phép tạo filter trang điểm từ ảnh khuôn mặt. Hệ thống sử dụng AI để nhận diện và trích xuất lớp trang điểm, sau đó tạo thành filter có thể áp dụng real-time trên camera.

## 🎯 Mục tiêu chính

- Tạo filter trang điểm từ ảnh có sẵn
- Áp dụng filter real-time trên camera
- Chỉnh sửa filter theo ý muốn
- Lưu trữ và quản lý filter theo user

## 🏗️ Kiến trúc hệ thống

```
Frontend (React/Next.js) ←→ Backend (FastAPI) ←→ AI/ML Services
                                    ↓
                            Storage (MinIO + PostgreSQL)
```

## 🚀 Tính năng chính

1. **Upload & Face Detection**: Upload ảnh, phát hiện và crop khuôn mặt
2. **Makeup Extraction**: Trích xuất lớp trang điểm từ khuôn mặt
3. **Filter Generation**: Tạo filter từ thông tin trang điểm
4. **Real-time Application**: Áp dụng filter real-time trên camera
5. **Filter Editing**: Chỉnh sửa filter theo từng vùng
6. **User Management**: Đăng ký, đăng nhập, lưu trữ dữ liệu

## 📁 Cấu trúc dự án

```
facetory/
├── frontend/          # React/Next.js application
├── backend/           # FastAPI server
├── ai_models/         # AI/ML models and services
├── docs/              # Documentation
├── docker-compose.yml # Docker configuration
└── README.md
```

## 🛠️ Công nghệ sử dụng

### Frontend
- React/Next.js
- TypeScript
- Tailwind CSS
- WebRTC (camera access)

### Backend
- FastAPI (Python)
- PostgreSQL
- Redis
- MinIO (object storage)

### AI/ML
- RetinaFace (face detection)
- U-Net (face segmentation)
- BeautyGAN (makeup transfer)
- MediaPipe (landmarks)

### Infrastructure
- Docker & Docker Compose
- Nginx (production)

## 📖 Tài liệu chi tiết

- [Yêu cầu hệ thống](./requirements.md)
- [Thiết kế hệ thống](./system-design.md)
- [User Flow](./user-flow.md)
- [API Documentation](./api-docs.md)
- [Deployment Guide](./deployment.md)

## 🚀 Quick Start

```bash
# Clone repository
git clone <repository-url>
cd facetory

# Start development environment
docker-compose up --build

# Access applications
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# MinIO Console: http://localhost:9001
```

## 📝 License

MIT License 