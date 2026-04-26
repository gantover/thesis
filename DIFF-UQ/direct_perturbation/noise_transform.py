import os
import glob
import argparse
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

def create_pixel_perturbations(exp_dir, m_samples=10, noise_scale=0.02, use_transforms=False):
    """
    Takes base images and creates M sets of perturbed images.
    Supports pure Gaussian noise or Mixed TTA (Transforms + Noise).
    """
    base_dir = os.path.join(exp_dir, "0", "imgs")
    
    # Verify the base directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base image directory not found: {base_dir}")
        
    image_paths = sorted(glob.glob(os.path.join(base_dir, "*.png")))
    if not image_paths:
        print(f"No images found in {base_dir}")
        return
        
    print(f"Found {len(image_paths)} base images.")
    print(f"Generating {m_samples} perturbed sets...")
    print(f" - Gaussian Noise Sigma: {noise_scale}")
    print(f" - Spatial Transforms Enabled: {use_transforms}")
    
    # Extract dimensions from the first image to dynamically set crop size
    sample_img = Image.open(image_paths[0])
    W, H = sample_img.size
    
    # Define the spatial TTA pipeline (used only if use_transforms=True)
    if use_transforms:
        spatial_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(size=(H, W), scale=(0.95, 1.0), antialias=True)
        ])
    
    # Loop over the M samples to create
    for m in range(1, m_samples + 1):
        target_dir = os.path.join(exp_dir, str(m), "imgs")
        os.makedirs(target_dir, exist_ok=True)
        
        for img_path in tqdm(image_paths, desc=f"Generating Set {m}/{m_samples}"):
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # 1. Apply spatial transforms (if enabled)
            if use_transforms:
                img = spatial_transforms(img)
                
            # Convert to PyTorch tensor [C, H, W] in range [0.0, 1.0]
            img_tensor = F.to_tensor(img)
            
            # 2. Add Gaussian noise
            noise = torch.randn_like(img_tensor) * noise_scale
            perturbed_tensor = img_tensor + noise
            
            # 3. Clip back to valid image range [0.0, 1.0]
            perturbed_tensor = torch.clamp(perturbed_tensor, 0.0, 1.0)
            
            # Convert back to PIL Image and save
            perturbed_img = F.to_pil_image(perturbed_tensor)
            
            filename = os.path.basename(img_path)
            save_path = os.path.join(target_dir, filename)
            perturbed_img.save(save_path)
            
    print(f"\nSuccess! Generated folders 1 through {m_samples} in {exp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pixel-level perturbations to test CLIP fragility.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to the main experiment directory")
    parser.add_argument("--m", type=int, default=10, help="Number of perturbation sets to generate (Default: 10)")
    parser.add_argument("--sigma", type=float, default=0.02, help="Scale of the Gaussian pixel noise (Default: 0.02)")
    parser.add_argument("--use_transforms", action="store_true", help="Enable Mixed TTA (Crop + Flip + Noise)")
    
    args = parser.parse_args()
    
    create_pixel_perturbations(
        args.exp_dir, 
        m_samples=args.m, 
        noise_scale=args.sigma, 
        use_transforms=args.use_transforms
    )