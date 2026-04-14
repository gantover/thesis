import os
import glob
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def create_pixel_perturbations(exp_dir, m_samples=10, noise_scale=0.02):
    """
    Takes the base images from exp_dir/0/imgs/ and creates M sets of 
    perturbed images in exp_dir/1/imgs/, exp_dir/2/imgs/, etc.
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
    print(f"Generating {m_samples} perturbed sets with Gaussian noise (sigma={noise_scale})...")
    
    # Loop over the M samples to create
    for m in range(1, m_samples + 1):
        target_dir = os.path.join(exp_dir, str(m), "imgs")
        os.makedirs(target_dir, exist_ok=True)
        
        for img_path in tqdm(image_paths, desc=f"Generating Set {m}/{m_samples}"):
            # Load image and convert to 0-1 float array
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Add Gaussian noise
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=img_array.shape)
            perturbed_array = img_array + noise
            
            # Clip back to valid image range [0, 1] and convert to uint8
            perturbed_array = np.clip(perturbed_array, 0.0, 1.0)
            perturbed_array = (perturbed_array * 255.0).astype(np.uint8)
            
            # Save the perturbed image with the exact same filename
            filename = os.path.basename(img_path)
            save_path = os.path.join(target_dir, filename)
            Image.fromarray(perturbed_array).save(save_path)
            
    print(f"\nSuccess! Generated folders 1 through {m_samples} in {exp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pixel-level perturbations to test CLIP fragility.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to the main experiment directory (e.g., ADM/results/imagenet/...)")
    parser.add_argument("--m", type=int, default=10, help="Number of perturbation sets to generate (Default: 10)")
    parser.add_argument("--sigma", type=float, default=0.02, help="Scale of the Gaussian pixel noise (Default: 0.02)")
    
    args = parser.parse_args()
    
    create_pixel_perturbations(args.exp_dir, m_samples=args.m, noise_scale=args.sigma)