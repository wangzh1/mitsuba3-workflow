from abc import ABC, abstractmethod
from tqdm import tqdm

class MitsubaTrainer(ABC):
    def __init__(self,
                 scene: mi.Scene,
                 params: mi.ParameterMap,
                 optimizer: mi.ad.Optimizer = None,
                 criterion: callable = None,
                 max_epochs: int = 100,
                 device: str = 'cpu'):
        """
        Args:
            scene (mi.Scene): Mitsuba scene to render.
            params (mi.ParameterMap): Differentiable parameters to optimize.
            optimizer (mi.ad.Optimizer): Optimizer for gradient-based updates.
            criterion (callable): Loss function to minimize; takes rendered image and target as input.
            max_epochs (int): Maximum number of training epochs.
            device (str): Device to use (e.g., 'cuda' or 'cpu').
        """
        self.scene = scene
        self.params = params
        self.optimizer = optimizer if optimizer else mi.ad.Adam(lr=0.05)
        self.criterion = criterion if criterion else self.default_criterion
        self.max_epochs = max_epochs
        self.device = device

        # Set Mitsuba variant based on device
        mi.set_variant(f'{device}_ad_rgb')

    @staticmethod
    def default_criterion(rendered_image, target_image):
        """Default criterion: Mean Squared Error (MSE)"""
        return dr.mean(dr.square(rendered_image - target_image))

    def render(self, spp: int = 128):
        """Renders the scene using Mitsuba."""
        return mi.render(self.scene, self.params, spp=spp)

    @abstractmethod
    def fitting_step(self, target_image, spp: int = 128):
        """
        Abstract method to be implemented by subclasses.
        Performs a single fitting step.
        Args:
            target_image (mi.TensorXf): Target image to match.
            spp (int): Sample per pixel for rendering.
        Returns:
            list[float]: List of loss values for the current step.
        """
        pass

    def fit(self, target_image, spp: int = 128):
        """
        Training loop to optimize scene parameters.

        Args:
            target_image (mi.TensorXf): Target image to match.
            spp (int): Sample per pixel for rendering.
        """
        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch + 1}/{self.max_epochs}")
            losses = []
            for _ in tqdm(range(spp), desc=f"Epoch {epoch + 1}"):
                losses.extend(self.fitting_step(target_image, spp=spp))
            avg_loss = sum(losses) / len(losses)
            print(f"Average Loss: {avg_loss:.6f}")

        print("Training complete.")

# Example usage:
# Define a subclass that implements fitting_step
class CustomMitsubaTrainer(MitsubaTrainer):
    def fitting_step(self, target_image, spp: int = 128):
        """
        Performs a single fitting step.
        Args:
            target_image (mi.TensorXf): Target image to match.
            spp (int): Sample per pixel for rendering.
        Returns:
            list[float]: List of loss values for the current step.
        """
        # Forward pass: render the image
        rendered_image = self.render(spp=spp)
        
        # Compute multiple losses (e.g., MSE, SSIM, etc.)
        loss_mse = self.criterion(rendered_image, target_image)
        # Add more losses here
        
        # Backward pass: compute gradients with respect to scene parameters
        dr.backward(loss_mse)
        
        # Update parameters
        self.optimizer.step()
        
        # Zero gradients for the next iteration
        self.optimizer.zero_grad()
        
        return [loss_mse.item()]

# Load scene and create trainer
scene = mi.load_file('../scenes/cbox.xml', res=128, integrator='prb')
params = mi.traverse(scene)
key = 'red.reflectance.value'
params[key] = mi.Color3f(0.01, 0.2, 0.9)
params.update()

trainer = CustomMitsubaTrainer(scene, params)

# Render reference image
image_ref = trainer.render(spp=512)
target_image = mi.TensorXf(image_ref)

# Start training
trainer.fit(target_image, spp=4)