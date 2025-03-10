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
                 lambda_total : int = 100,
                 learning_rate: float = 1,
                 max_stages: int = 1,
                 max_iterations: int = 1000,
                 val_interval: int = 100,
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
        import mitransient as mitr
        if scene_path is None:
            raise ValueError("Scene path must be provided!")
        self.scene_path = scene_path
        self.scene = mi.load_file(self.scene_path)
        
        self.gt = self.init_ground_truth(self.scene)

        self.params = mi.traverse(self.scene)
        self.initial_vertex_pos = dr.unravel(mi.Point3f, self.params['light.vertex_positions'])
        self.keys_to_optimize = ['trans']
        self.optimizer = mi.ad.Adam(lr=learning_rate, params={'trans': mi.Point2f(50.0, 0.0)})
        
        # criterion
        self.criterion = criterion
        self.lambda_total = lambda_total

        # training settings
        self.max_stages = max_stages
        self.max_iterations = max_iterations
        self.val_interval = val_interval
        self.device = device

    @staticmethod
    def init_ground_truth(scene) -> Dict[str, Any]:
        """
        Initialize the ground truth image.
        """
        image_ref, transient_ref = mi.render(scene, spp=1024)
        return {'image_ref': image_ref,
                'transient_ref': transient_ref}
    
    @staticmethod
    def apply_transformation(params, opt, initial_vertex_pos):
        trafo = mi.Transform4f().translate([opt['trans'].x, opt['trans'].y, 0.0])
        opt['trans'].y = dr.clip(opt['trans'].y, 0, 1)
        params['light.vertex_positions'] = dr.ravel(trafo @ initial_vertex_pos)

    def fitting_step(self, idx):
        self.apply_transformation(self.params, self.optimizer, self.initial_vertex_pos)

        image, transient = mi.render(self.scene, self.params, spp=16)
    
        # Evaluate the objective function from the current rendered image
        loss = self.criterion(transient, self.gt['transient_ref']) * self.lambda_total
        
        return {'loss': loss, 
                'image_vis': image}