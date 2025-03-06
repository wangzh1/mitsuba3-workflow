import time
from typing import Any, Dict, List
from abc import ABC, abstractmethod

import drjit as dr
import mitsuba as mi
import numpy as np
import tqdm
import wandb


class MitsubaTrainer(ABC):
    def __init__(self,
                 scene: mi.Scene,
                 params: mi.SceneParameters,
                 optimizer: mi.ad.Adam = None,
                 criterion: callable = None,
                 max_epochs: int = 100,
                 device: str = 'cuda'):
        """
        Args:
            scene (mi.Scene): Mitsuba scene to render.
            params (mi.ParameterMap or dict): Differentiable parameters to optimize.
            optimizer (torch.optim.Optimizer): Optimizer for gradient-based updates.
            criterion (callable): Loss function to minimize; takes rendered image and target as input.
            max_epochs (int): Maximum number of training epochs.
            lr_scheduler (optional): Learning rate scheduler.
            device (str): Device to use (e.g., 'cuda' or 'cpu').
        """
        self.scene = scene
        self.params = params
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.device = device

        # Initialize Mitsuba rendering
        mi.set_variant('cuda_ad_rgb' if device == 'cuda' else 'llvm_ad_rgb')
    @abstractmethod
    def render(self):
        """Renders the scene using Mitsuba."""
        image = mi.render(self.scene, self.params)
        return image

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


    def fit(self, target_image):
        """
        Training loop to optimize scene parameters.

        Args:
            target_image (mi.TensorXf): Target image to match.
        """
        for epoch in range(self.max_epochs):
            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass: render the image
            rendered_image = self.render()

            # Compute loss
            loss = self.criterion(rendered_image, target_image)

            # Backward pass: compute gradients with respect to scene parameters
            dr.backward(loss)

            # Update parameters
            self.optimizer.step()

        print("Training complete.")

    def save_params(self, filename):
        """Saves optimized parameters to a file."""
        self.params.write(filename)

    def load_params(self, filename):
        """Loads parameters from a file."""
        self.params.read(filename)
