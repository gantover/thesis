import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

class ImagePerturber:
    """
    Takes a base image and applies on-the-fly perturbations (noise + transforms).
    """
    def __init__(self, m_samples: int = 10, noise_scale: float = 0.02, use_transforms: bool = False):
        self.m_samples = m_samples
        self.noise_scale = noise_scale
        self.use_transforms = use_transforms
        
    def __call__(self, img: Image.Image) -> list[Image.Image]:
        """
        Returns a list of `m_samples` images, where the FIRST image is always the 
        untouched base image, matching Laplace '0' directory behavior.
        """
        W, H = img.size
        
        spatial_transforms = None
        if self.use_transforms:
            spatial_transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                # T.RandomResizedCrop(size=(H, W), scale=(0.95, 1.0), antialias=True)
            ])
            
        # First sample is the original unperturbed base image
        perturbed_images = [img.copy()]
        
        # M-1 subsequent samples are perturbed
        for _ in range(self.m_samples - 1):
            current_img = img.copy()
            if spatial_transforms:
                current_img = spatial_transforms(current_img)
                
            img_tensor = TF.to_tensor(current_img)
            
            # Add Gaussian noise
            if self.noise_scale > 0:
                noise = torch.randn_like(img_tensor) * self.noise_scale
                perturbed_tensor = img_tensor + noise
                perturbed_tensor = torch.clamp(perturbed_tensor, 0.0, 1.0)
            else:
                perturbed_tensor = img_tensor
                
            perturbed_img = TF.to_pil_image(perturbed_tensor)
            perturbed_images.append(perturbed_img)
            
        return perturbed_images

