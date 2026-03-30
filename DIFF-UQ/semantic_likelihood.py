import os
from PIL import Image
import argparse
from pathlib import Path

import torch
import numpy as np
import clip


def parse_args():
    parser = argparse.ArgumentParser(description="Generative Uncertainty Calculation")
    parser.add_argument("--path", type=str, required=True, help="Path to the samples")
    parser.add_argument("--M", type=int, required=False, default=5 + 1, help="Number of MC samples")
    parser.add_argument(
        "--encoder",
        type=str,
        default="clip",
        choices=["clip", "dinov2_vits14_reg"],
        help="Encoder used to extract image features",
    )
    return parser.parse_args()


def gaussian_entropy(mu_array: np.ndarray, sigma_squared: float) -> np.ndarray:
    """
    Calculate the entropy of multivariate Gaussian distributions with covariance
    Diag(1/M * Σ(μₘ²) - μ̄²) + σ²I in batch mode.
    """
    if len(mu_array.shape) == 2:
        mu_array = mu_array[np.newaxis, ...]

    _, _, D = mu_array.shape

    diagonal_terms = np.mean(mu_array**2, axis=1) - np.mean(mu_array, axis=1) ** 2
    diagonal_terms = np.clip(diagonal_terms, 0.0, None)  # because with only M=6 samples there are some negative values
    eigenvalues = diagonal_terms + sigma_squared  # Shape: (N, D)
    log_det = np.sum(np.log(eigenvalues), axis=1)  # Shape: (N,)

    entropy = 0.5 * log_det + 0.5 * D * (np.log(2 * np.pi) + 1)

    if len(mu_array.shape) == 2:
        return entropy[0]
    return entropy


def dinov2_preprocess(image: Image.Image) -> torch.Tensor:
    image = image.resize((256, 256), resample=Image.BICUBIC)
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    image = np.asarray(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    return image


def build_encoder(encoder_name: str, device: torch.device):
    if encoder_name == "clip":
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            return model.encode_image(image_batch)

        return preprocess, encode_batch

    if encoder_name == "dinov2_vits14_reg":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        model.eval().to(device)
        preprocess = dinov2_preprocess

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            features = model.forward_features(image_batch)
            patch_tokens = features["x_norm_patchtokens"]
            # Aggregate clean patch tokens into a single semantic embedding per image.
            return patch_tokens.mean(dim=1)

        return preprocess, encode_batch

    raise ValueError(f"Unknown encoder: {encoder_name}")


def compute_generative_uncertainty(path, M, eu_type="entropy", encoder_name="clip"):
    print(f"Loading samples from {M} models from {path} using encoder '{encoder_name}'")

    #### 1) Compute image features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess, encode_batch = build_encoder(encoder_name, device)

    # count number of images in path
    N = len(os.listdir(f"{path}/{0}/imgs"))
    feature_filename = f"{encoder_name}_features.pt"

    for m in range(M):
        print(f"Processing model {m}")
        image_vecs = []
        for i in range(N):
            image = Image.open(f"{path}/{m}/imgs/{i:05d}.png").convert("RGB")
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_vecs.append(encode_batch(image).cpu())

        image_vecs = torch.concat(image_vecs, dim=0)
        print(image_vecs.shape)
        features_path = Path(path) / str(m) / feature_filename
        features_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(image_vecs, features_path)

    #### 2) Compute the entropy of the semantic likelihood

    features = []
    for m in range(M):
        path_m = Path(path) / str(m) / feature_filename
        features.append(torch.load(path_m, map_location="cpu"))

    features = torch.stack(features, dim=0)
    features = np.transpose(features.cpu().numpy(), (1, 0, 2))
    print(features.shape)

    if eu_type == "entropy":
        eu = gaussian_entropy(features, sigma_squared=1e-3)
    else:
        raise ValueError(f"Unknown epistemic uncertainty type: {eu_type}")

    print(f"Saving: {path}/{eu_type}_{encoder_name}.npy")
    eu_path = Path(path) / f"{eu_type}_{encoder_name}.npy"
    eu_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(eu_path, eu)


if __name__ == "__main__":
    args = parse_args()
    compute_generative_uncertainty(args.path, args.M, encoder_name=args.encoder)
