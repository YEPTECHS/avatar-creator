from fastapi import APIRouter, Depends

from utils import get_models


router = APIRouter(prefix="/avatar", tags=["avatar"])


@router.post("/create")
async def create_avatar(models: Depends(get_models)):
    pass