from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import JSONResponse

from core.flows import create_avatar_flow
from utils import get_models

router = APIRouter(prefix="/avatar", tags=["avatar"])


@router.post("/create")
async def create_avatar(video_file: UploadFile, models: Depends(get_models)):
    avatar = create_avatar_flow(models, video_file)
    return JSONResponse(status_code=200, content={"message": f"Created avatar: {avatar.name}"})
    