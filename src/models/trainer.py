import mitsuba as mi
import drjit as dr
import numpy as np
import tqdm
import time


class MitsubaTrainer:
    def __init__(self,
                 scene: mi.Scene,
                 params: mi.ParameterMap,
                 optimizer: mi.ad.Adam = None,
                 criterion: callable = None,
                 max_epochs: int = 100,
                 lr_scheduler: ,
                 device='cuda'):
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
        self.lr_scheduler = lr_scheduler
        self.device = device

        # Initialize Mitsuba rendering
        mi.set_variant('cuda_ad_rgb' if device == 'cuda' else 'llvm_ad_rgb')

    def render(self):
        """Renders the scene using Mitsuba."""
        image = mi.render(self.scene, self.params)
        return image

    def train(self, target_image):
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

            # Update learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step(loss)

            # Log progress
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.max_epochs}], Loss: {loss.item():.4f}")

        print("Training complete.")

    def save_params(self, filename):
        """Saves optimized parameters to a file."""
        self.params.write(filename)

    def load_params(self, filename):
        """Loads parameters from a file."""
        self.params.read(filename)


# Example usage
if __name__ == "__main__":
    # Load Mitsuba scene
    scene = mi.load_file("scene.xml")

    # Define differentiable parameters (e.g., material reflectance, light intensity)
    params = mi.traverse(scene)
    param_key = "material.reflectance.value"
    dr.enable_grad(params[param_key])

    # Define optimizer (e.g., Adam)
    optimizer = mi.ad.Adam(lr=0.01)
    optimizer.set_params(params)

    # Define loss function (e.g., L2 loss between rendered and target image)
    def l2_loss(rendered: mi.TensorXf,
                target: mi.TensorXf) -> mi.TensorXf:
        return dr.sum(dr.sqr(rendered - target))

    # Define target image (e.g., pre-rendered or synthetic)
    # Replace with actual target image
    target_image = mi.TensorXf(np.random.rand(256, 256, 3))

    # Initialize trainer
    trainer = MitsubaTrainer(
        scene=scene,
        params=params,
        optimizer=optimizer,
        criterion=l2_loss,
        max_epochs=100,
        device='cuda'
    )

    # Train
    trainer.train(target_image)

    # Save optimized parameters
    trainer.save_params("optimized_params.xml")
