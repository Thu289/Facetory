# Hướng dẫn xem logs

## Xem logs backend (tất cả)
```bash
docker compose logs backend
```

## Xem logs backend (realtime - follow)
```bash
docker compose logs -f backend
```

## Xem logs backend (số dòng cuối cùng)
```bash
# Xem 100 dòng cuối
docker compose logs --tail=100 backend

# Xem 300 dòng cuối (để xem đầy đủ quá trình xử lý)
docker compose logs --tail=300 backend
```

## Xem logs backend và filter theo keywords

### Filter theo BiSeNet logs:
```bash
docker compose logs backend | grep -E "📊|RAW BISENET|BISENET RESULTS"
```

### Filter theo resize logs:
```bash
docker compose logs backend | grep -E "🔄|RESIZE|TRƯỚC|SAU"
```

### Filter theo mask creation logs:
```bash
docker compose logs backend | grep -E "📝|TẠO.*MASK|MASK CREATED"
```

### Filter tất cả logs quan trọng:
```bash
docker compose logs backend | grep -E "📊|🔄|✅|❌|⚠️|📝|RAW|RESIZE|MASK|BISENET"
```

## Xem logs realtime với filter
```bash
# Theo dõi logs realtime và chỉ hiển thị logs quan trọng
docker compose logs -f backend 2>&1 | grep --line-buffered -E "📊|🔄|✅|❌|⚠️|📝|RAW|RESIZE|MASK|BISENET"
```

## Xem logs và tìm lỗi cụ thể

### Tìm warnings:
```bash
docker compose logs backend | grep -E "⚠️|WARNING"
```

### Tìm errors:
```bash
docker compose logs backend | grep -E "❌|ERROR|Error|Exception"
```

### Tìm logs về pixel counts:
```bash
docker compose logs backend | grep -E "pixels|pixel"
```

### Tìm logs về class IDs:
```bash
docker compose logs backend | grep -E "class_id|class_ID"
```

## Xem logs từ nhiều services
```bash
# Xem logs từ cả backend và frontend
docker compose logs backend frontend

# Xem logs từ tất cả services
docker compose logs
```

## Lưu logs vào file
```bash
# Lưu logs vào file
docker compose logs backend > backend_logs.txt

# Lưu logs với timestamp
docker compose logs backend > backend_logs_$(date +%Y%m%d_%H%M%S).txt

# Lưu logs đã filter
docker compose logs backend | grep -E "📊|🔄|✅|❌|RAW|RESIZE|MASK" > filtered_logs.txt
```

## Xem logs trong Docker container (nếu cần)
```bash
# Vào container
docker exec -it facetory-backend-1 bash

# Xem logs trực tiếp trong container
tail -f /var/log/app.log  # (nếu có file log)
# hoặc logs sẽ hiển thị qua stdout của uvicorn
```

## Xem logs với timestamps
```bash
# Hiển thị timestamps
docker compose logs -t backend

# Hiển thị timestamps và follow
docker compose logs -f -t backend
```

## Ví dụ: Xem logs khi test filter
```bash
# Terminal 1: Theo dõi logs realtime
docker compose logs -f backend 2>&1 | grep --line-buffered -E "📊|🔄|✅|RAW|RESIZE|MASK|BISENET"

# Terminal 2: Test filter trên frontend (http://localhost:3000/filter)
# Sau đó quay lại Terminal 1 để xem logs chi tiết
```

## Log sections quan trọng để debug

Khi test filter, tìm các sections sau trong logs:

1. **📊 RAW BISENET RESULTS**: Kết quả raw từ BiSeNet
2. **🔄 BEFORE RESIZE / AFTER RESIZE**: Trước và sau khi resize
3. **📝 TẠO [REGION] MASK**: Quá trình tạo từng mask
4. **🔄 RESIZE [REGION] MASK**: Resize từng mask
5. **✅ / ❌**: Kết quả thành công hoặc thất bại

