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

    async def download_file(self, object_name: str, local_path: str = None):
        """
        Download file from MinIO
        
        Args:
            object_name: Object name in bucket
            local_path: Optional local path to save file. If None, returns bytes.
        
        Returns:
            bytes if local_path is None, otherwise None
        """
        try:
            if local_path:
                # Save to file
                self.client.fget_object(
                    self.bucket_name,
                    object_name,
                    local_path
                )
                return None
            else:
                # Return bytes
                response = self.client.get_object(self.bucket_name, object_name)
                data = response.read()
                response.close()
                response.release_conn()
                return data
        except Exception as e:
            raise Exception(f"Failed to download file: {e}")

    async def delete_file(self, object_name: str):
        try:
            self.client.remove_object(self.bucket_name, object_name)
        except Exception as e:
            raise Exception(f"Failed to delete file: {e}")

    def get_file_url(self, object_name: str, expires: int = 3600, use_proxy: bool = True) -> str:
        """
        Get file URL - either presigned MinIO URL or proxy URL
        
        Args:
            object_name: Object name in bucket
            expires: Expiration time in seconds (default 1 hour)
            use_proxy: If True, return proxy URL instead of direct MinIO URL
        """
        if use_proxy:
            # Return proxy URL that goes through backend
            from urllib.parse import quote
            encoded_object = quote(object_name, safe='')
            return f"/api/makeup/storage/file/{encoded_object}"
        else:
            try:
                from datetime import timedelta
                # Convert seconds to timedelta
                expires_delta = timedelta(seconds=expires)
                url = self.client.presigned_get_object(
                    self.bucket_name,
                    object_name,
                    expires=expires_delta
                )
                # Replace container hostname with localhost for browser access
                url = url.replace("minio:9000", "localhost:9000")
                return url
            except Exception as e:
                raise Exception(f"Failed to get file URL: {e}") 