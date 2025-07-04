# Facetory - AI Face Filter System

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### Development Setup

1. **Clone and setup project**
```bash
git clone <repository-url>
cd facetory
```

2. **Start all services**
```bash
docker-compose up --build
```

3. **Access applications**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001 (admin/admin)

### Development Commands

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild and start
docker-compose up --build

# Clean up volumes
docker-compose down -v
```

## 📁 Project Structure

```
facetory/
├── frontend/              # React/Next.js application
│   ├── app/              # Next.js app directory
│   ├── components/       # React components
│   ├── Dockerfile        # Frontend container
│   └── package.json      # Frontend dependencies
├── backend/              # FastAPI application
│   ├── app/              # Backend source code
│   │   ├── api/          # API endpoints
│   │   ├── core/         # Configuration
│   │   └── services/     # Business logic
│   ├── Dockerfile        # Backend container
│   └── requirements.txt  # Python dependencies
├── docs/                 # Documentation
├── docker-compose.yml    # Docker orchestration
└── README.md            # This file
```

## 🎯 Current Status

### ✅ Phase 1.1: Project Setup (COMPLETED)
- [x] Docker environment
- [x] Frontend (React/Next.js)
- [x] Backend (FastAPI)
- [x] Database (PostgreSQL)
- [x] Object storage (MinIO)
- [x] Cache (Redis)

### ✅ Phase 1.2: Basic Upload (COMPLETED)
- [x] Upload component with drag & drop
- [x] File validation (type, size)
- [x] Image preview
- [x] Error handling

### 🔄 Phase 1.3: Face Detection (IN PROGRESS)
- [ ] Integrate RetinaFace model
- [ ] Face detection API
- [ ] Display detection results

### ⏳ Phase 1.4: Face Cropping (PENDING)
- [ ] Crop tool component
- [ ] Multiple face selection
- [ ] Crop processing API

## 🧪 Testing

### Frontend Testing
```bash
cd frontend
npm run dev
# Access at http://localhost:3000
```

### Backend Testing
```bash
cd backend
uvicorn main:app --reload
# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### API Testing
```bash
# Test upload endpoint
curl -X POST "http://localhost:8000/api/upload/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-image.jpg"
```

## 📚 Documentation

- [System Requirements](./docs/requirements.md)
- [System Design](./docs/system-design.md)
- [User Flow](./docs/user-flow.md)
- [Development Roadmap](./docs/development-roadmap.md)

## 🛠️ Development

### Adding New Features
1. Check the [Development Roadmap](./docs/development-roadmap.md)
2. Follow the phase structure
3. Test each feature independently
4. Update documentation

### Code Style
- Frontend: ESLint + Prettier
- Backend: Black + isort
- TypeScript for frontend
- Type hints for Python

## 🚀 Deployment

### Production Setup
```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Variables
Create `.env` file for production:
```env
DATABASE_URL=postgresql://user:pass@host:5432/db
MINIO_URL=https://your-minio.com
JWT_SECRET=your-production-secret
```

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review the roadmap
3. Create an issue in the repository

## 📄 License

MIT License - see LICENSE file for details 