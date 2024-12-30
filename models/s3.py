from pydantic import BaseModel, Field


class S3DownloadParams(BaseModel):
    s3_bucket: str = Field(..., description="S3 bucket name")
    s3_base_object_path: str = Field(..., description="S3 object base path")
    s3_region: str = Field(..., description="S3 region")
    local_dir: str = Field(..., description="local path")