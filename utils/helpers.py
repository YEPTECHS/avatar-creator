from fastapi import Request


async def get_models(request: Request):
    return request.app.state.models
