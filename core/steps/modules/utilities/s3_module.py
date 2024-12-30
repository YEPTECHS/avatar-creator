import aioboto3
import os

from prefect import task
from pathlib import Path
from typing import List, Optional
from models.errors import S3DownloadError, S3ListError
from models.s3 import S3DownloadParams


class S3Module:
    def __init__(self, s3_params: S3DownloadParams):
        self.bucket = s3_params.s3_bucket
        self.base_dir = s3_params.s3_base_object_path.rstrip('/')
        self.region = s3_params.s3_region
        self.local_dir = Path(s3_params.local_dir)
        self.session = aioboto3.Session()

    @task
    async def download_to_local(self, filename: str) -> str:
        """
        Asynchronously download a file from S3 to the local directory.
        
        Args:
            filename (str): File name to download
            
        Returns:
            str: Local file path
            
        Raises:
            S3DownloadError: If download fails
        """
        s3_path = f"{self.base_dir}/{filename}"
        local_path = self.local_dir / filename
        
        try:
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            async with self.session.client('s3', region_name=self.region) as s3_client:
                await s3_client.download_file(
                    Bucket=self.bucket,
                    Key=s3_path,
                    Filename=str(local_path)
                )
            
            return str(local_path)
        except Exception as e:
            raise S3DownloadError(f"Failed to download file: {str(e)}", s3_path)

    @task
    async def upload_to_s3(self, local_path: str, s3_filename: Optional[str] = None) -> str:
        """
        Asynchronously upload a file to S3.
        
        Args:
            local_path (str): Local file path to upload
            s3_filename (Optional[str]): Custom filename for S3 (defaults to local filename)
            
        Returns:
            str: S3 path of uploaded file
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        s3_filename = s3_filename or local_path.name
        s3_path = f"{self.base_dir}/{s3_filename}"
        
        try:
            async with self.session.client('s3', region_name=self.region) as s3_client:
                await s3_client.upload_file(
                    Filename=str(local_path),
                    Bucket=self.bucket,
                    Key=s3_path
                )
            return s3_path
        except Exception as e:
            raise S3DownloadError(f"Failed to upload file: {str(e)}", s3_path)

    @task
    async def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """
        List files in S3 bucket with optional prefix.
        
        Args:
            prefix (Optional[str]): Prefix to filter files
            
        Returns:
            List[str]: List of file names
        """
        try:
            full_prefix = f"{self.base_dir}/{prefix if prefix else ''}"
            async with self.session.client('s3', region_name=self.region) as s3_client:
                paginator = s3_client.get_paginator('list_objects_v2')
                files = []
                
                async for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                    if 'Contents' in page:
                        files.extend([obj['Key'] for obj in page['Contents']])
                
                # Remove base_dir prefix from results
                return [f.replace(f"{self.base_dir}/", "") for f in files]
        except Exception as e:
            raise S3ListError(str(e), self.bucket, self.base_dir)

    @task
    async def batch_download(self, filenames: List[str]) -> List[str]:
        """
        Download multiple files from S3.
        
        Args:
            filenames (List[str]): List of filenames to download
            
        Returns:
            List[str]: List of local file paths
        """
        local_paths = []
        for filename in filenames:
            local_path = await self.download_to_local(filename)
            local_paths.append(local_path)
        return local_paths

    @task
    async def batch_upload(self, local_paths: List[str], s3_filenames: Optional[List[str]] = None) -> List[str]:
        """
        Upload multiple files to S3.
        
        Args:
            local_paths (List[str]): List of local file paths
            s3_filenames (Optional[List[str]]): Optional list of custom S3 filenames
            
        Returns:
            List[str]: List of S3 paths
        """
        s3_paths = []
        s3_filenames = s3_filenames or [None] * len(local_paths)
        
        for local_path, s3_filename in zip(local_paths, s3_filenames):
            s3_path = await self.upload_to_s3(local_path, s3_filename)
            s3_paths.append(s3_path)
        return s3_paths

    @task
    async def get_latest_version(self, prefix: Optional[str] = None) -> Optional[str]:
        """
        Get the latest version file from S3 bucket.
        
        Args:
            prefix (Optional[str]): Prefix to filter files
            
        Returns:
            Optional[str]: Latest version file name or None if no files found
        """
        try:
            full_prefix = f"{self.base_dir}/{prefix if prefix else ''}"
            async with self.session.client('s3', region_name=self.region) as s3_client:
                paginator = s3_client.get_paginator('list_objects_v2')
                latest_file = None
                latest_time = None
                
                async for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if latest_time is None or obj['LastModified'] > latest_time:
                                latest_time = obj['LastModified']
                                latest_file = obj['Key']
                
                if latest_file:
                    return latest_file.replace(f"{self.base_dir}/", "")
                return None
                
        except Exception as e:
            raise S3ListError(str(e), self.bucket, self.base_dir)

    @task
    async def download_latest(self, prefix: Optional[str] = None) -> Optional[str]:
        """
        Download the latest version file from S3.
        
        Args:
            prefix (Optional[str]): Prefix to filter files
            
        Returns:
            Optional[str]: Local path of downloaded file or None if no files found
        """
        latest_file = await self.get_latest_version(prefix)
        if latest_file:
            return await self.download_to_local(latest_file)
        return None

    @task
    async def download_version(self, model_name: str, version: str) -> Optional[str]:
        """
        Download specific version of model file from S3.
        
        Args:
            model_name (str): Name of the model
            version (str): Version number
            
        Returns:
            Optional[str]: Local path of downloaded file or None if not found
        """
        try:
            # List files in version directory
            files = await self.list_files()
            
            # Filter for model files with specific version
            version_prefix = f"v{version}"
            model_files = [f for f in files if f.startswith(version_prefix)]
            
            if not model_files:
                return None
            
            # Get latest file if multiple exist
            latest_file = sorted(model_files)[-1]
            
            # Download the file
            local_path = await self.download_to_local(latest_file)
            return local_path
            
        except Exception as e:
            raise S3DownloadError(f"Failed to download version {version} of {model_name}: {str(e)}", 
                                f"{self.base_dir}/{version_prefix}")
