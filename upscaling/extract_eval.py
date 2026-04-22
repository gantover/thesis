import os
import random
import click
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop
from tqdm import tqdm

@click.command()
@click.option("--data_dir", default="/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/val", help="Path to ImageNet val or test directory")
@click.option("--out_dir", default="./eval_dataset", help="Base output directory")
@click.option("--num_samples", default=500, help="Number of image pairs to extract")
def extract_eval(data_dir, out_dir, num_samples):
    """Extracts paired 64x64 and 256x256 images from a dataset."""
    
    # Setup directories
    low_res_dir = os.path.join(out_dir, "low_res")
    high_res_dir = os.path.join(out_dir, "high_res")
    os.makedirs(low_res_dir, exist_ok=True)
    os.makedirs(high_res_dir, exist_ok=True)

    # Find all images recursively (handles both flat folders and class-subfolders)
    print("Scanning directory for images...")
    valid_extensions = {".jpg", ".jpeg", ".png"}
    all_images = [
        p for p in Path(data_dir).rglob("*") 
        if p.suffix.lower() in valid_extensions
    ]
    
    if not all_images:
        print("No images found! Check your data_dir.")
        return

    # Pick a random subset
    selected_images = random.sample(all_images, min(num_samples, len(all_images)))
    print(f"Extracting {len(selected_images)} pairs...")

    # Setup transforms
    # We use CenterCrop for validation to ensure deterministic evaluation
    high_res_transform = Compose([
        Resize(256, interpolation=Image.BICUBIC),
        CenterCrop(256)
    ])
    
    low_res_transform = Compose([
        Resize(64, interpolation=Image.BICUBIC)
    ])

    # Process and save
    for idx, img_path in enumerate(tqdm(selected_images)):
        try:
            # Ensure image is RGB (some ImageNet images are grayscale)
            img = Image.open(img_path).convert("RGB")
            
            # Create 256x256 Ground Truth
            hr_img = high_res_transform(img)
            
            # Create 64x64 Input Condition (downsampled from the HR crop to perfectly align)
            lr_img = low_res_transform(hr_img)
            
            # Save them with a clean index name (e.g., 0000.png, 0001.png)
            filename = f"{idx:04d}.png"
            hr_img.save(os.path.join(high_res_dir, filename))
            lr_img.save(os.path.join(low_res_dir, filename))
            
        except Exception as e:
            print(f"Skipped {img_path.name} due to error: {e}")

    print(f"\nDone! Paired images saved to: {out_dir}")

if __name__ == "__main__":
    extract_eval()