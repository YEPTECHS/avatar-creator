import uvloop
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from core.routers.health import router as health_router
from core.routers.avatar import router as avatar_router
from utils import setup_logger
from settings import settings
from models.errors import S3ListError, S3DownloadError, UnzipError, DataValidationError, CudaError
from utils import (http_exception_handler,
                   validation_exception_handler,
                   generic_exception_handler,
                   s3_download_exception_handler,
                   unzip_exception_handler,
                   s3_list_exception_handler,
                   data_build_exception_handler,
                   runtime_exception_handler,
                   cuda_exception_handler)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = setup_logger(name="avatar")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # TODO: prepare all models
        # app.state.models = prepare()
        logger.info("App initialization completed.")
        yield
    except Exception as e:
        logger.error(f"Error: exception during startup: {e}")
        raise
    finally:
        try:
            logger.info("Application is shutting down...")
        except Exception as e:
            logger.error(f"Error: exception during shutdown: {e}")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.BASE_PREFIX)
app.include_router(avatar_router, prefix=settings.BASE_PREFIX)

# HTTP exception handler
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
app.add_exception_handler(S3DownloadError, s3_download_exception_handler)
app.add_exception_handler(UnzipError, unzip_exception_handler)
app.add_exception_handler(S3ListError, s3_list_exception_handler)
app.add_exception_handler(DataValidationError, data_build_exception_handler)
app.add_exception_handler(RuntimeError, runtime_exception_handler)
app.add_exception_handler(CudaError, cuda_exception_handler)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)