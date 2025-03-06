import time
from typing import Optional

import drjit as dr
import mitsuba as mi
import tqdm
from omegaconf import DictConfig

from models.trainer_base import MitsubaTrainer


class TransientTrainer(MitsubaTrainer):
    def __init__(self,
                 scene: mi.Scene,
                 params: mi.SceneParameters,
                 optimizer: mi.ad.Adam = None,
                 criterion: callable = None,
                 max_epochs: int = 100,
                 device: str = 'cuda'):
        super().__init__(scene, params, optimizer, criterion, max_epochs, device)

    def fit(self, target_image):
        with tqdm(range(self.max_epochs), desc="Fitting") as pbar:
            for epoch in pbar:
                start_time = time.time()
                self.optimizer.zero_grad()
                rendered_image = self.render()
                loss = self.criterion(rendered_image, target_image)
                dr.backward(loss)

                self.optimizer.step()
                end_time = time.time()
                epoch_time = end_time - start_time
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
