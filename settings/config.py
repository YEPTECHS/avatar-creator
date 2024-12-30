from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import Field


class Settings(BaseSettings):
    LOG_LEVEL: str = Field(default="DEBUG", description="Log level")
    BASE_PREFIX: str = Field(default="/api/v1", description="Base prefix")
    
    # aws config
    AWS_ACCESS_KEY_ID: str = Field(..., description="AWS access key id")
    AWS_SECRET_KEY: str = Field(..., description="AWS secret key")
    AWS_REGION: str = Field(..., description="AWS region")
    AVATARS_S3_BUCKET_NAME: str = Field(..., description="S3 bucket name")
    AVATARS_S3_OBJECT_BASE_NAME: str = Field(..., description="S3 object base name")
    MODELS_S3_BUCKET_NAME: str = Field(..., description="S3 bucket name")
    MODELS_S3_OBJECT_BASE_NAME: str = Field(..., description="S3 object base name")  


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()