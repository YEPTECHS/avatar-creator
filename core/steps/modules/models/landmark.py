import asyncio
import numpy as np

from typing import Any, List, Tuple
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
from prefect import task

from .base import BaseModelModule
from core.steps.utils.face_detection import FaceAlignment, LandmarksType
from core.steps.utils.face_parsing import FaceParsing


class LandmarkModule(BaseModelModule):
    @task
    def __init__(self, model_name: str, device: str, config_file: str = "core/steps/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py", concurrent_per_model: int = 2, version: str = 'latest'):
        """
        Args:
            model_name (str): The name of the model
            device (str): The device to load model on (e.g. 'cuda:0', 'cpu')
            config_file (str): Path to the MMPose config file
            concurrent_per_model (int): Maximum number of concurrent tasks per model
            version (str): Model version to use
        """
        super().__init__(model_name, device, concurrent_per_model, version)
        
        # TODO: make this configurable
        self.config_file = config_file
        self.face_alignment = FaceAlignment(LandmarksType._2D, flip_input=False, device=self.device)
        self.face_parsing = FaceParsing(device=self.device)

    @task
    async def _load_model_on_device(self, device: str, local_model_checkpoint_path: str, **kwargs) -> Any:
        """Load MMPose model on specified device"""
        if not local_model_checkpoint_path or not self.config_file:
            raise ValueError("Model checkpoint and config file must be initialized")

        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            None,
            init_model,
            self.config_file,
            local_model_checkpoint_path,
            device
        )
        return model

    @task
    def forward(self, model: Any, frames: List[np.ndarray], upperbondrange=0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        batches = [frames[i:i + 1] for i in range(0, len(frames), 1)]
        coord_placeholder = (0.0,0.0,0.0,0.0)
        coords_list = []
        average_range_minus = []
        average_range_plus = []

        for frame in batches:
            results = inference_topdown(model, np.asarray([frame])[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_landmark: np.ndarray = keypoints[0][23:91]
            face_landmark = face_landmark.astype(np.int32)
            bbox = self.face_alignment.get_detections_for_batch(np.asarray(frame))
            for _, f in enumerate(bbox):
                if f is None: # no face in the image
                    coords_list += [coord_placeholder]
                    continue
                
                half_face_coord =  face_landmark[29]
                range_minus = (face_landmark[30]- face_landmark[29])[1]
                range_plus = (face_landmark[29]- face_landmark[28])[1]
                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)
                if upperbondrange != 0:
                    half_face_coord[1] = upperbondrange+half_face_coord[1]
                half_face_dist = np.max(face_landmark[:,1]) - half_face_coord[1]
                upper_bond = half_face_coord[1]-half_face_dist
                
                f_landmark = (np.min(face_landmark[:, 0]),int(upper_bond),np.max(face_landmark[:, 0]),np.max(face_landmark[:,1]))
                x1, y1, x2, y2 = f_landmark
                
                if y2-y1<=0 or x2-x1<=0 or x1<0:
                    coords_list += [f]
                    _w, _h = f[2]-f[0], f[3]-f[1]
                    self.logger.error(f"error bbox: {f}")
                else:
                    coords_list += [f_landmark]
            
        return coords_list, frames