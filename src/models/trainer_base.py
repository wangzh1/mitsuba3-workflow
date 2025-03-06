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
                 scene: mi.Scene,
                 params: mi.SceneParameters,
                 optimizer: mi.ad.Adam = None,
                 criterion: Callable = None,
                 max_stages: int = 1,
                 max_iterations: int = 500,
                 device: str = 'cuda'):
        """
        Args:
            scene (mi.Scene): Mitsuba scene to render.
            params (mi.ParameterMap or dict): Differentiable parameters to optimize.
            optimizer (torch.optim.Optimizer): Optimizer for gradient-based updates.
            criterion (Callable): Loss function to minimize; takes rendered image and target as input.
            max_epochs (int): Maximum number of training epochs.
            max_iterations (int): Maximum number of iterations per epoch.
            device (str): Device to use (e.g., 'cuda' or 'cpu').
        """
        self.scene = scene
        self.params = params
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_stages = max_stages
        self.max_iterations = max_iterations
        self.device = device
        
        self.gt = self.init_ground_truth()
    
        # Initialize Mitsuba rendering
        mi.set_variant('cuda_ad_rgb' if device == 'cuda' else 'llvm_ad_rgb')

    @abstractmethod
    def init_ground_truth(self) -> Any:
        """
        Initializes the ground truth for the optimization process.
        """
        raise NotImplementedError

    @abstractmethod
    def fitting_step(self, target_image) -> Dict[str, Any]:
        """
        Performs a single fitting step.
        Args:
            target_image (mi.TensorXf): Target image to match.
        Returns:
            float: Loss value for the current step.
        """
        raise NotImplementedError

    @abstractmethod
    def on_stage_start(self):
        pass

    @abstractmethod
    def on_stage_end(self):
        pass

    def fit(self):
        """
        Training loop to optimize scene parameters.
        """
                # Outer progress bar for stages
        stage_pbar = tqdm(range(self.max_stages), desc="Stages", unit="stage", position=0)

        for stage in stage_pbar:
            self.on_stage_start()
            # Inner progress bar for iterations
            iter_pbar = tqdm(range(self.max_iterations), desc="Iterations", unit="iter", position=1, leave=False)

            for iter in iter_pbar:
                # Zero gradients
                self.optimizer.zero_grad()
                rendered_image = self.render()

                loss = self.fitting_step(rendered_image)
                dr.backward(loss)

                self.optimizer.step()

                iter_pbar.set_description(f"Iteration {iter + 1}/{self.max_iterations}, Loss: {loss:.4f}")
            
            self.on_stage_end()
            # Update stage progress bar
            stage_pbar.set_description(f"Stage {stage + 1}/{self.max_stages} completed")

    def save_params(self, filename):
        """Saves optimized parameters to a file."""
        self.params.write(filename)

    def load_params(self, filename):
        """Loads parameters from a file."""
        self.params.read(filename)
