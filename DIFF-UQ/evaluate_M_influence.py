"""
Evaluate the influence of the number of Monte Carlo samples M on generative
uncertainty estimates.

Given a pre-existing experiment directory that was run with mc_size=M_max
(containing subdirectories 0 .. M_max for the MAP + M_max Laplace samples),
this script reuses those images and re-computes the CLIP-based entropy for
every prefix M ∈ {M_values}, without having to re-generate any images.

Usage
-----
    python evaluate_M_influence.py \
        --path /path/to/exp_dir \
        --M_max 5 \
        --M_values 1 2 3 4 5

The script then saves:
    {path}/entropy_clip_M{m}.npy   for each m in M_values
and produces a summary plot:
    {path}/M_influence.png
"""

import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import clip


# ---------------------------------------------------------------------------
# Helpers (kept in sync with semantic_likelihood.py)
# ---------------------------------------------------------------------------

def gaussian_entropy(mu_array: np.ndarray, sigma_squared: float) -> np.ndarray:
    """
    Entropy of the moment-matched Gaussian over M CLIP embeddings.

    mu_array : (N, M, D)  – N images, M models, D CLIP dims
    """
    if len(mu_array.shape) == 2:
        mu_array = mu_array[np.newaxis, ...]

    _, _, D = mu_array.shape

    diagonal_terms = np.mean(mu_array ** 2, axis=1) - np.mean(mu_array, axis=1) ** 2
    diagonal_terms = np.clip(diagonal_terms, 0.0, None)
    eigenvalues = diagonal_terms + sigma_squared          # (N, D)
    log_det = np.sum(np.log(eigenvalues), axis=1)         # (N,)

    entropy = 0.5 * log_det + 0.5 * D * (np.log(2 * np.pi) + 1)
    return entropy


def encode_or_load_clip_features(path: str, m: int, device: str,
                                 model, preprocess,
                                 force_recompute: bool = False) -> torch.Tensor:
    """
    Load cached CLIP features for model index m, or compute and cache them.
    """
    cache_path = f"{path}/{m}/clip_features.pt"

    if os.path.exists(cache_path) and not force_recompute:
        return torch.load(cache_path)

    imgs_dir = f"{path}/{m}/imgs"
    N = len([f for f in os.listdir(imgs_dir) if f.endswith(".png")])
    clip_vecs = []
    for i in range(N):
        image = preprocess(
            Image.open(f"{imgs_dir}/{i:05d}.png")
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_vecs.append(model.encode_image(image))

    feats = torch.concat(clip_vecs, dim=0)
    torch.save(feats, cache_path)
    print(f"  Saved CLIP features → {cache_path}")
    return feats


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def evaluate_M_influence(path: str, M_max: int, M_values: list[int],
                         sigma_squared: float = 1e-3,
                         force_recompute_clip: bool = False) -> dict:
    """
    For each m in M_values, compute the entropy using subdirectories 0 .. m
    (i.e. the MAP model at index 0 plus m Laplace samples).

    Returns a dict  {m: entropy_array (N,)}
    """
    # Validate
    for m in M_values:
        assert 1 <= m <= M_max, f"M={m} out of range [1, {M_max}]"
        for s in range(m + 1):
            imgs_dir = f"{path}/{s}/imgs"
            assert os.path.isdir(imgs_dir), (
                f"Missing expected subdirectory: {imgs_dir}\n"
                f"Make sure the experiment was run with mc_size >= {M_max}."
            )

    # Load CLIP model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device} …")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Encode / load features for every subdirectory we need
    max_s = max(M_values)
    all_features: dict[int, torch.Tensor] = {}
    for s in range(max_s + 1):
        print(f"CLIP features – model {s}/{max_s}")
        all_features[s] = encode_or_load_clip_features(
            path, s, device, clip_model, preprocess,
            force_recompute=force_recompute_clip,
        )

    # Compute entropy for each M value
    results: dict[int, np.ndarray] = {}
    for m in sorted(M_values):
        # stacked shape: (m+1, N, D)  then transposed → (N, m+1, D)
        feats = torch.stack([all_features[s] for s in range(m + 1)], dim=0)
        feats_np = np.transpose(feats.cpu().numpy(), (1, 0, 2))  # (N, m+1, D)

        eu = gaussian_entropy(feats_np, sigma_squared=sigma_squared)
        results[m] = eu

        save_path = f"{path}/entropy_clip_M{m}.npy"
        np.save(save_path, eu)
        print(f"  M={m}: mean entropy={eu.mean():.4f}  std={eu.std():.4f}  → {save_path}")

    return results


def plot_results(results: dict[int, np.ndarray], save_path: str) -> None:
    """
    Two-panel figure:
      Left  – box-plot of per-image entropy distributions for each M
      Right – mean ± std of entropy vs. M
    """
    M_sorted = sorted(results.keys())
    entropies = [results[m] for m in M_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Box plot ---
    axes[0].boxplot(entropies, labels=M_sorted, notch=False, showfliers=False)
    axes[0].set_xlabel("M  (number of Laplace MC samples)")
    axes[0].set_ylabel("Entropy (nats)")
    axes[0].set_title("Distribution of generative uncertainty per M")
    axes[0].grid(axis="y", alpha=0.4)

    # --- Mean ± std ---
    means = [e.mean() for e in entropies]
    stds  = [e.std()  for e in entropies]
    axes[1].errorbar(M_sorted, means, yerr=stds, marker="o", capsize=4,
                     linewidth=1.5, color="steelblue")
    axes[1].set_xlabel("M  (number of Laplace MC samples)")
    axes[1].set_ylabel("Mean entropy (nats)")
    axes[1].set_title("Mean generative uncertainty vs. M")
    axes[1].set_xticks(M_sorted)
    axes[1].grid(alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved → {save_path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Study the influence of M (MC samples) on generative uncertainty"
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Path to the experiment directory (must contain subdirs 0 .. M_max)",
    )
    parser.add_argument(
        "--M_max", type=int, default=5,
        help="Maximum M used when the experiment was generated (default: 5)",
    )
    parser.add_argument(
        "--M_values", type=int, nargs="+", default=None,
        help="Which M values to evaluate (default: 1 .. M_max)",
    )
    parser.add_argument(
        "--sigma_squared", type=float, default=1e-3,
        help="σ² additive to diagonal of covariance (default: 1e-3)",
    )
    parser.add_argument(
        "--force_recompute_clip", action="store_true",
        help="Re-encode CLIP features even if cached .pt files exist",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    M_values = args.M_values if args.M_values is not None else list(range(1, args.M_max + 1))

    print(f"Evaluating M ∈ {M_values} in {args.path}")
    results = evaluate_M_influence(
        path=args.path,
        M_max=args.M_max,
        M_values=M_values,
        sigma_squared=args.sigma_squared,
        force_recompute_clip=args.force_recompute_clip,
    )

    plot_path = os.path.join(args.path, "M_influence.png")
    plot_results(results, save_path=plot_path)
