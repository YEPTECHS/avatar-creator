from typing import Tuple

from prefect import task

from settings import settings
from .modules import LandmarkModule, VAEModule


@task
def prepare_models() -> Tuple[LandmarkModule, VAEModule]:  
    # Landmark model
    # TODO: make device configurable
    landmark_module = LandmarkModule(model_name='mmpose', device='cuda:0')
    landmark_module.load_model()

    # VAE model
    vae_module = VAEModule(model_name='vae', device='cuda:0')
    vae_module.load_model()

    return landmark_module, vae_module
