# Quick Start Guide - Facetory Makeup Filter System

## 🚀 Get Started in 5 Minutes

### 1. Start Services
```bash
# Start all Docker containers
docker compose up -d

# Verify all services are running
docker ps

# Expected services:
# - facetory-backend-1 (port 8000)
# - facetory-frontend-1 (port 3000)
# - facetory-minio-1 (port 9000-9001)
# - facetory-postgres-1 (port 5432)
# - facetory-redis-1 (port 6379)
```

### 2. Access the Application

**Frontend**: http://localhost:3000
- **Home Page**: Style analysis and upload
- **Filter Page**: http://localhost:3000/filter - Real-time makeup filter

**Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs - Interactive Swagger UI
- **Health Check**: http://localhost:8000/health

**MinIO Dashboard**: http://localhost:9000
- Username: `minioadmin`
- Password: `minioadmin`

### 3. Create Your First Style

1. Open http://localhost:3000/filter
2. Click **"Create New Style"**
3. Upload an image with makeup
4. Wait ~10-30 seconds for processing
5. Style created with LUTs and shaders!

### 4. Apply Real-Time Filter

1. In the filter page, select the style you created
2. Click to start camera
3. Grant browser camera permissions
4. **Enjoy real-time makeup filter!** 🎉
5. Adjust intensity with the slider

---

## 📋 System Endpoints

### Main Pages
- `/` - Home page (style analysis)
- `/filter` - Real-time filter application

### Backend APIs
- `POST /api/makeup/style/create_complete` - Create style
- `GET /api/makeup/style/{style_id}` - Get style
- `POST /api/face/makeup/style_extract` - Extract style only

---

## ✅ System Status

- ✅ Backend: Running on port 8000
- ✅ Frontend: Running on port 3000
- ✅ MinIO: Running on port 9000
- ✅ PostgreSQL: Running on port 5432
- ✅ Redis: Running on port 6379

---

## 🐛 Quick Troubleshooting

**"Camera access denied"**
- Grant browser permissions
- Use HTTPS in production

**"BiSeNet failed"**
```bash
docker exec -it facetory-backend-1 bash
cd /app/ai_models/BiseNet
bash setup_bisenet.sh
```

**"Cannot access frontend"**
```bash
# Check frontend container
docker logs facetory-frontend-1

# Rebuild if needed
docker compose up -d --build frontend
```

**"WebGL not supported"**
- Update browser
- Check GPU drivers
- Visit `chrome://gpu` (Chrome)

---

## 📚 Full Documentation

- [System Specification](docs/makeup_filter_system.md)
- [Phase 1 Complete](docs/PHASE1_COMPLETE.md)
- [Phase 2 Complete](docs/PHASE2_COMPLETE.md)
- [System Complete](docs/SYSTEM_COMPLETE.md)
- [API Reference](API_URLS.md)

---

**Ready to create beautiful makeup filters!** ✨

