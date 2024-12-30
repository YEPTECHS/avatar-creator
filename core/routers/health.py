from fastapi import APIRouter
from fastapi.responses import JSONResponse

from utils import setup_logger


router = APIRouter(prefix="/health", tags=["health"])
logger = setup_logger(name="avatar.health")


@router.get('/ready', response_class=JSONResponse)
async def ready():
    return JSONResponse({'status': 'OK'}, headers={'Access-Control-Allow-Origin': '*'})
