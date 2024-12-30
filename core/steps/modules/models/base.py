import asyncio

from typing import Dict, Any
from prefect import task
from threading import Lock
from pathlib import Path

from utils import setup_logger
from core.steps.modules.utilities.s3_module import S3Module, S3DownloadParams
from settings import settings


class BaseModelModule:
    @task
    def __init__(self, model_name: str, device: str, concurrent_per_model: int = 4, version: str = 'latest'):
        """
        Args:
            model_name (str): The name of the model.
            device (str): The device to load model on (e.g. 'cuda:0', 'cpu').
            concurrent_per_model (int): The maximum number of concurrent tasks per model.
            version (str): Model version to use. Can be 'latest' or a version number.
        """
        self.model_name = model_name
        self.device = device
        self.concurrent_per_model = concurrent_per_model
        self.version = version

        self.logger = setup_logger(self.__class__.__name__)
        
        # model manager with semaphore
        self.model_info = None
        
        # Initialize model files
        self.local_model_dir = Path("/app/models")
        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        self.local_model_checkpoint_path = None
        
        # Download model files on initialization
        asyncio.create_task(self._init_model_files())

    @task
    async def load_model(self, **kwargs):
        """Load model instance on specified device."""
        model = await self._load_model_on_device(self.device, self.local_model_checkpoint_path, **kwargs)
        if model:
            self.model_info = {
                'model': model,
                'concurrent_count': 0,
                'lock': Lock(),
                'sem': asyncio.Semaphore(self.concurrent_per_model)
            }
        else:
            raise RuntimeError("Model could not be loaded")

    @task
    async def _load_model_on_device(self, device: str, local_model_checkpoint_path: str, **kwargs) -> Any:
        """
        Load a single model instance on specified device.
        Subclass must implement this method.
        
        Args:
            device: Target device
            **kwargs: Model specific parameters
        Returns:
            Loaded model instance
        """
        raise NotImplementedError("Subclass must implement _load_model_on_device method")

    @task
    def forward(self, model: Any, **kwargs) -> Any:
        raise NotImplementedError("Subclass must implement forward method")

    @task
    def _run_inference(self, model_info: Dict, **kwargs) -> Any:
        """Run inference in thread"""
        try:
            return self.forward(model_info['model'], **kwargs)
        finally:
            with model_info['lock']:
                model_info['concurrent_count'] = max(0, model_info['concurrent_count'] - 1)

    @task
    async def __call__(self, **kwargs) -> Any:
        # Check if model is loaded
        if not self.model_info:
            raise RuntimeError("No model loaded. Call load_model first.")
            
        try:
            # Use model-specific semaphore to control concurrency
            async with self.model_info['sem']:
                # Run inference in a separate thread
                result = await asyncio.to_thread(
                    self._run_inference,
                    self.model_info,
                    **kwargs
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            raise

    @task
    def process(self, **kwargs):
        """Sync interface"""
        return asyncio.run(self.__call__(**kwargs))

    @task
    async def download_model(self, s3_params: S3DownloadParams, version: str = 'latest') -> str:
        """
        Download model file from S3.
        
        Args:
            s3_params (S3DownloadParams): S3 download parameters
            version (str): Model version to download. Can be 'latest' or version number.
            
        Returns:
            str: Local path of downloaded model file
        """
        s3_module = S3Module(s3_params=s3_params)
        
        # Download model file based on version
        if version == 'latest':
            local_path = await s3_module.download_latest(self.model_name)
        else:
            local_path = await s3_module.download_version(self.model_name, version)
        
        if not local_path:
            raise ValueError(f"No model file found for {self.model_name} version {version}")
        
        # Verify file extension
        model_extensions = ('.pth', '.pt', '.bin', '.safetensors')
        if not any(local_path.endswith(ext) for ext in model_extensions):
            raise ValueError(f"Downloaded file is not a valid model file: {local_path}")
        
        return local_path

    @task
    async def _init_model_files(self) -> None:
        """Initialize model file by downloading from S3"""
        model_name_lower = self.model_name.lower()
        version_prefix = "latest" if self.version == "latest" else f"v{self.version}"
        
        s3_params = S3DownloadParams(
            s3_bucket=settings.MODELS_S3_BUCKET_NAME,
            s3_base_object_path=f"{settings.MODELS_S3_OBJECT_BASE_NAME}/{model_name_lower}/{version_prefix}",
            s3_region=settings.AWS_REGION,
            local_dir=str(self.local_model_dir / model_name_lower / version_prefix)
        )
        
        try:
            model_path = await self.download_model(s3_params, version=self.version)
            self.logger.info(f"Successfully downloaded model file for {self.model_name} version {version_prefix}")
            self.local_model_checkpoint_path = model_path
        except Exception as e:
            self.logger.error(f"Failed to download model file for {self.model_name} version {version_prefix}: {str(e)}")
            raise