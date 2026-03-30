#!/usr/bin/env python3
"""Compute HPSv3 human preference scores for generated images.

Scores are saved as hps.npy (float32, shape [n_images]) and hps_sigma.npy
in the parent of --img-dir (same location as realism.npy / rarity.npy).

Class labels are reproduced deterministically from the generation seed,
matching the exact RNG sequence used in ADM/main.py:
  seed_everything(seed) -> torch.randn (fixed_xT) -> torch.randint (fixed_classes)

Image i in {s}/imgs/ was conditioned on class:
  fixed_classes[i % batch_size, i // batch_size]
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reproduce_classes(seed, total_n_sample, channels, image_size, batch_size):
    """Reproduce the exact class assignments from ADM/main.py's RNG sequence.

    Replicates:
        seed_everything(seed)
        fixed_xT = torch.randn([total_n_sample, C, H, W])   # advances RNG
        fixed_classes = torch.randint(0, 1000, (batch_size, n_rounds))
    """
    seed_everything(seed)
    torch.randn([total_n_sample, channels, image_size, image_size])
    n_rounds = total_n_sample // batch_size
    return torch.randint(0, 1000, (batch_size, n_rounds))


def get_imagenet_labels():
    """Return list of 1000 ImageNet class names using torchvision metadata."""
    from torchvision.models import ResNet50_Weights
    return ResNet50_Weights.DEFAULT.meta["categories"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img-dir", type=str, required=True,
                        help="Path to {s}/imgs/ directory of generated images")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed used during generation (default: 1234)")
    parser.add_argument("--total-n-sample", type=int, default=12032,
                        help="Total number of samples generated (default: 12032)")
    parser.add_argument("--sample-batch-size", type=int, default=256,
                        help="Batch size used during generation (default: 256)")
    parser.add_argument("--fixed-class", type=int, default=10000,
                        help="Fixed class used during generation; 10000 = random classes (default: 10000)")
    parser.add_argument("--channels", type=int, default=3,
                        help="Number of image channels used during generation (default: 3)")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Spatial image size used during generation (default: 128)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for hps.npy (default: parent of --img-dir)")
    parser.add_argument("--hps-batch-size", type=int, default=8,
                        help="Batch size for HPSv3 inference (default: 8; 7B model is VRAM-limited)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for HPSv3 inference (default: cuda)")
    parser.add_argument("--classes-npy", type=str, default=None,
                        help="Path to classes.npy saved by main.py (shape: [batch_size, n_rounds]). "
                             "If provided, skips RNG reproduction.")
    return parser.parse_args()


def main():
    args = parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir) if args.out_dir else img_dir.parent

    # Collect images in sorted order — matches ImageFolderDataset alphabetical sort
    image_paths = sorted(img_dir.glob("*.png"))
    n_images = len(image_paths)
    if n_images == 0:
        raise RuntimeError(f"No PNG files found in {img_dir}")
    print(f"Found {n_images} images in {img_dir}")

    # --- Resolve per-image class labels ---
    labels = get_imagenet_labels()

    if args.classes_npy is not None:
        # Use pre-saved classes.npy (shape: [batch_size, n_rounds])
        fixed_classes = torch.from_numpy(np.load(args.classes_npy))
        class_indices = [
            fixed_classes[i % fixed_classes.shape[0], i // fixed_classes.shape[0]].item()
            for i in range(n_images)
        ]
        print(f"Loaded class assignments from {args.classes_npy}")
    elif args.fixed_class != 10000:
        # All images conditioned on the same class
        class_indices = [args.fixed_class] * n_images
        print(f"Using fixed class {args.fixed_class}: {labels[args.fixed_class]}")
    else:
        # Reproduce exact class assignments by replaying the generation RNG sequence
        print(f"Reproducing class assignments (seed={args.seed}, "
              f"total_n_sample={args.total_n_sample}, batch_size={args.sample_batch_size})...")
        fixed_classes = reproduce_classes(
            seed=args.seed,
            total_n_sample=args.total_n_sample,
            channels=args.channels,
            image_size=args.image_size,
            batch_size=args.sample_batch_size,
        )
        class_indices = [
            fixed_classes[i % args.sample_batch_size, i // args.sample_batch_size].item()
            for i in range(n_images)
        ]

    prompts = [f"a high quality photo of a {labels[c]}" for c in class_indices]

    # --- Resume support ---
    out_mu = out_dir / "hps.npy"
    out_sigma = out_dir / "hps_sigma.npy"
    mu_values = []
    sigma_values = []
    start_idx = 0

    if out_mu.exists():
        existing_mu = np.load(out_mu)
        start_idx = len(existing_mu)
        mu_values = list(existing_mu)
        sigma_values = list(np.load(out_sigma)) if out_sigma.exists() else [0.0] * start_idx
        print(f"Resuming from image {start_idx}/{n_images}")

    if start_idx >= n_images:
        print("All images already scored.")
        return

    # --- HPSv3 inference ---
    from hpsv3 import HPSv3RewardInferencer
    print(f"Loading HPSv3 model on {args.device}...")
    inferencer = HPSv3RewardInferencer(device=args.device)

    batch_size = args.hps_batch_size
    remaining_paths = [str(p) for p in image_paths[start_idx:]]
    remaining_prompts = prompts[start_idx:]

    for batch_start in tqdm(range(0, len(remaining_paths), batch_size), desc="HPSv3 scoring"):
        batch_paths = remaining_paths[batch_start : batch_start + batch_size]
        batch_prompts = remaining_prompts[batch_start : batch_start + batch_size]

        rewards = inferencer.reward(batch_paths, batch_prompts)
        for reward in rewards:
            mu_values.append(reward[0].item())
            sigma_values.append(reward[1].item())

        # Incremental save so cluster job interruptions lose at most one batch
        np.save(out_mu, np.array(mu_values, dtype=np.float32))
        np.save(out_sigma, np.array(sigma_values, dtype=np.float32))

    mu_arr = np.array(mu_values, dtype=np.float32)
    print(f"\nSaved {out_mu} and {out_sigma}")
    print(f"Score stats: mean={mu_arr.mean():.4f}  std={mu_arr.std():.4f}  "
          f"min={mu_arr.min():.4f}  max={mu_arr.max():.4f}")


if __name__ == "__main__":
    main()
