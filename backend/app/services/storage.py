from minio import Minio
from app.core.config import settings

class MinioService:
    def __init__(self):
        self.client = Minio(
            settings.MINIO_URL.replace("http://", ""),
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False  # Set to True for HTTPS
        )
        self.bucket_name = settings.MINIO_BUCKET
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except Exception as e:
            print(f"Error ensuring bucket exists: {e}")

    async def upload_file(self, local_path: str, object_name: str) -> str:
        try:
            self.client.fput_object(
                self.bucket_name,
                object_name,
                local_path
            )
            return f"{self.bucket_name}/{object_name}"
        except Exception as e:
            raise Exception(f"Failed to upload file: {e}")

    async def download_file(self, object_name: str, local_path: str):
        try:
            self.client.fget_object(
                self.bucket_name,
                object_name,
                local_path
            )
        except Exception as e:
            raise Exception(f"Failed to download file: {e}")

    async def delete_file(self, object_name: str):
        try:
            self.client.remove_object(self.bucket_name, object_name)
        except Exception as e:
            raise Exception(f"Failed to delete file: {e}")

    def get_file_url(self, object_name: str, expires: int = 3600) -> str:
        try:
            return self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=expires
            )
        except Exception as e:
            raise Exception(f"Failed to get file URL: {e}") 