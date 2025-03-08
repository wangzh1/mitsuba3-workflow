import time
from typing import Any, Callable, Dict, Optional

import drjit as dr
import mitsuba as mi
import tqdm
from omegaconf import DictConfig

from src.models.misc.criterion import MSE
from src.models.trainer_base import MitsubaTrainer


class TransientTrainer(MitsubaTrainer):
    def __init__(self,
                 scene_path: str = None,
                 criterion: Callable = MSE(),
                 learning_rate: float = 0.1,
                 max_stages: int = 1,
                 max_iterations: int = 500,
                 val_interval: int = 1,
                 device: str = 'cuda'):
        """
        Args:
            scene_path: Path to the scene to be trained.
            criterion: Criterion to be used for training.
            learning_rate: Learning rate for the optimizer.
            max_stages: Maximum number of stages for training.
            max_iterations: Maximum number of iterations for each stage.
            val_interval: Interval for validation.
            device: Device to be used for training.
        """
        mi.set_variant('cuda_ad_rgb' if device == 'cuda' else 'llvm_ad_rgb')
        if scene_path is None:
            raise ValueError("Scene path must be provided!")
        self.scene_path = scene_path
        self.scene = mi.load_file(self.scene_path, res=128, integrator='prb')
        
        self.gt = self.init_ground_truth(self.scene)

        self.params = mi.traverse(self.scene)
        self.keys_to_optimize = ['red.reflectance.value']
        self.params['red.reflectance.value'] = mi.Color3f(0.01, 0.2, 0.9)
        self.optimizer = mi.ad.Adam(lr=learning_rate, params={key: self.params[key] for key in self.keys_to_optimize})
        self.params.update(self.optimizer)
        self.criterion = criterion
        self.max_stages = max_stages
        self.max_iterations = max_iterations
        self.val_interval = val_interval
        self.device = device
        # Initialize Mitsuba rendering

    @staticmethod
    def init_ground_truth(scene) -> Dict[str, Any]:
        """
        Initialize the ground truth image.
        """
        image_ref = mi.render(scene, spp=512)
        return {'image_ref': image_ref}
    
    def fitting_step(self):
        image = mi.render(self.scene, self.params, spp=4)
    
        # Evaluate the objective function from the current rendered image
        loss = self.criterion(image, self.gt['image_ref'])
        
        return loss
