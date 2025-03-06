import time
from typing import Optional, Callable

import drjit as dr
import mitsuba as mi
import tqdm
from omegaconf import DictConfig

from models.trainer_base import MitsubaTrainer
from models.misc import MSE


class ImageTrainer(MitsubaTrainer):
    def __init__(self,
                 scene: mi.Scene = None,
                 params: mi.SceneParameters = None,
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
        self.criterion = MSE()
        self.max_stages = max_stages
        self.max_iterations = max_iterations
        self.device = device
        
        self.gt = self.init_ground_truth()

        # Initialize Mitsuba rendering
        mi.set_variant('cuda_ad_rgb' if device == 'cuda' else 'llvm_ad_rgb')

    def init_ground_truth(self):
        """
        Initialize the ground truth image.
        """
        scene = mi.load_file('../scenes/cbox.xml', res=128, integrator='prb')
        params = mi.traverse(scene)

        key = 'red.reflectance.value'

        # Save the original value
        param_ref = mi.Color3f(params[key])

        # Set another color value and update the scene
        params[key] = mi.Color3f(0.01, 0.2, 0.9)
        
        params.update()
    def fitting_step(self, target_image):
        pass
