from typing import Tuple

from prefect import flow

from core.steps.modules import LandmarkModule, VAEModule


@flow
def prepare_models_flow() -> Tuple[LandmarkModule, VAEModule]:  
    # Landmark model
    # TODO: make device configurable
    landmark_module = LandmarkModule(model_name='mmpose', device='cuda:0')
    landmark_module.load_model()

    # VAE model
    vae_module = VAEModule(model_name='madebyollin/taesd', device='cuda:0')
    vae_module.load_model()

    return landmark_module, vae_module
