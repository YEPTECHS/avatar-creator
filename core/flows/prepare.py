from prefect import flow

from steps.model_prepare import prepare_models


@flow
def prepare(video_path: str):
    '''This flow is responsible for preparing all the models used in the avatar creator.'''
    landmark_module, vae_module = prepare_models()
    
    return 
