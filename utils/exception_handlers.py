from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from models.errors import S3DownloadError, UnzipError, S3ListError, DataValidationError, CudaError
from .logging import setup_logger


logger = setup_logger(name="avatar.exception.handler")


async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"HTTP error: {exc.detail}"}
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"message": "Validation error", "details": exc.errors()}
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("An unexpected error occurred")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred", "detail": str(exc)}
    )


async def runtime_exception_handler(request: Request, exc: RuntimeError):
    logger.error(f"Runtime error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "Runtime error", "details": str(exc)}
    )


async def s3_download_exception_handler(request: Request, exc: S3DownloadError):
    logger.error(f"S3 download error: {exc.message} - S3 path: {exc.s3_path}")
    return JSONResponse(
        status_code=500,
        content={"message": f"S3 download error: {exc.message}", "s3_path": exc.s3_path},
    )


async def unzip_exception_handler(request: Request, exc: UnzipError):
    logger.error(f"Unzip error: {exc.message} - Zip path: {exc.zip_path}")
    return JSONResponse(
        status_code=500,
        content={"message": f"Unzip error: {exc.message}", "zip_path": exc.zip_path},
    )


async def s3_list_exception_handler(request: Request, exc: S3ListError):
    logger.error(f"S3 list error: {exc.message} - Bucket: {exc.s3_bucket}, Base path: {exc.s3_base_object_path}")
    return JSONResponse(
        status_code=500,
        content={
            "message": f"S3 list error: {exc.message}",
            "s3_bucket": exc.s3_bucket,
            "s3_base_object_path": exc.s3_base_object_path
        },
    )


async def data_build_exception_handler(request: Request, exc: DataValidationError):
    logger.error(f"Data class building error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"message": f"Data class building error, {exc.message}"}
    )
    

async def cuda_exception_handler(request: Request, exc: CudaError):
    logger.error(f"Cuda initializing error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"message": f"Cuda initializing error, {exc.message}"}
    )