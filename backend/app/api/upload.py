from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
from datetime import datetime
from typing import Optional

from app.core.config import settings
from app.services.storage import MinioService

router = APIRouter()

@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file
    """
    # Validate file type
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Validate file size
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}{file_extension}"
        
        # Save file locally first
        local_path = os.path.join(settings.UPLOAD_DIR, filename)
        with open(local_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # TODO: Upload to MinIO
        # minio_service = MinioService()
        # minio_path = await minio_service.upload_file(local_path, filename)
        
        return JSONResponse({
            "success": True,
            "filename": filename,
            "local_path": local_path,
            "size": file.size,
            "message": "Image uploaded successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/status/{filename}")
async def get_upload_status(filename: str):
    """
    Get upload status for a file
    """
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    file_size = os.path.getsize(file_path)
    
    return {
        "filename": filename,
        "exists": True,
        "size": file_size,
        "uploaded_at": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
    } 