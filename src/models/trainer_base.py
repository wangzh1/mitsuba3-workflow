import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import drjit as dr
import mitsuba as mi
import numpy as np
import wandb
from tqdm import tqdm


class MitsubaTrainer(ABC):
    def __init__(self,
                 scene_path: str = None,
                 criterion: Callable = None,
                 learning_rate: float = 0.1,
                 max_stages: int = 1,
                 max_iterations: int = 500,
                 val_interval: int = 100,
                 device: str = 'cuda',
                 manual_optmize: bool = False):
        """
        Args:
            scene_path (str): Path to the Mitsuba scene file.
            criterion (Callable): Loss function to minimize; takes rendered image and target as input.
            learning_rate (float): Learning rate for the optimizer.
            max_stages (int): Maximum number of training stages.
            max_iterations (int): Maximum number of iterations per stage.
            val_interval (int): Interval for validation.
            device (str): Device to use (e.g., 'cuda' or 'cpu').
        """
        self.scene_path = scene_path
        self.scene = mi.load_file(self.scene_path, res=128, integrator='prb')
        self.gt = self.init_ground_truth(self.scene)

        self.params = mi.traverse(self.scene)
        self.keys_to_optimize = ['red.reflectance.value']
        self.optimizer = mi.ad.Adam(lr=learning_rate, prams={key: self.params[key] for key in self.keys_to_optimize})
        self.params.update(self.optimizer)
        self.criterion = criterion
        self.max_stages = max_stages
        self.max_iterations = max_iterations
        self.val_interval = val_interval
        self.device = device
        self.manual_optmize = manual_optmize
        # Initialize Mitsuba rendering
        mi.set_variant('cuda_ad_rgb' if device == 'cuda' else 'llvm_ad_rgb')

    @staticmethod
    def init_ground_truth(scene) -> Dict[str, Any]:
        """
        Initialize the ground truth image.
        """
        raise NotImplementedError

    @abstractmethod
    def fitting_step(self) -> Dict[str, Any]:
        """
        Performs a single fitting step.
        Args:
        Returns:
            float: Loss value for the current step.
        """
        raise NotImplementedError

    def on_stage_start(self, stage_idx):
        pass

    def on_stage_end(self, stage_idx):
        pass

    def on_iter_start(self):
        pass

    def on_iter_end(self):
        pass

    def fit(self):
        """
        Training loop to optimize scene parameters.
        """
        # Outer progress bar for stages
        stage_pbar = tqdm(range(self.max_stages), desc="Stages", unit="stage", position=0)

        for stage_idx, stage in enumerate(stage_pbar):
            self.on_stage_start(stage_idx)
            # Inner progress bar for iterations
            iter_pbar = tqdm(range(self.max_iterations), desc="Iterations", unit="iter", position=1, leave=False)

            for iter_idx, iter in enumerate(iter_pbar):
                self.on_iter_start()
                iter_result = self.fitting_step(iter_idx)
                loss = iter_result['loss']
                if not self.manual_optmize:
                    dr.backward(loss)
                    self.optimizer.step()
                    # Post-process the optimized parameters to ensure legal color values.
                    # for key in self.keys_to_optimize:
                    #     self.params[key] = dr.clip(self.params[key], 0.0, 1.0)
                    self.params.update(self.optimizer)
            
                iter_pbar.set_description(f"Iteration {iter}/{self.max_iterations}, Loss: {loss}")
                if iter_idx % self.val_interval == 0:
                    # import pdb; pdb.set_trace()
                    image_vis = iter_result['image_vis']
                    wandb.log({"stage": stage_idx, "step": iter_idx, "val_image": wandb.Image(image_vis), "loss": np.array(loss)})
                self.on_iter_end()
            
            self.on_stage_end(stage_idx)
            # Update stage progress bar
            stage_pbar.set_description(f"Stage {stage_idx + 1}/{self.max_stages} completed")
