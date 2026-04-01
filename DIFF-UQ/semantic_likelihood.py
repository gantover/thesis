import os
from PIL import Image
import argparse
from pathlib import Path
import sys
import warnings
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import clip
import open_clip

from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(description="Generative Uncertainty Calculation")
    parser.add_argument("--path", type=str, required=True, help="Path to the samples")
    parser.add_argument("--M", type=int, required=False, default=5 + 1, help="Number of MC samples")
    parser.add_argument(
        "--encoder",
        type=str,
        default="clip",
        choices=["clip", "dinov2_vits14_reg", "vgg16", "siglip", "openclip_h14"],
        help="Encoder used to extract image features",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of images per chunk when computing uncertainty from saved features",
    )
    return parser.parse_args()

def exact_gaussian_entropy(mu_array: np.ndarray, sigma_squared: float = 1e-3) -> np.ndarray:
    """
    Computes the EXACT full-covariance entropy of a multivariate Gaussian 
    using the Dual Covariance (Gram Matrix) trick to avoid dimensional explosion.
    """
    # np.linalg.eigvalsh does not support float16; promote to float64 for stability.
    mu_array = np.asarray(mu_array, dtype=np.float64)

    if len(mu_array.shape) == 2:
        mu_array = mu_array[np.newaxis, ...]

    B, M, D = mu_array.shape
    entropy = np.empty(B, dtype=np.float64)

    # The number of dimensions where true variance actually exists is at most M-1
    active_dims = M - 1
    
    # The constant term for the full D-dimensional entropy
    constant_term = 0.5 * D * (np.log(2 * np.pi) + 1)

    for i in range(B):
        samples = mu_array[i]  # Shape: (M, D)
        
        # 1. Center the M samples
        mean = np.mean(samples, axis=0)
        centered = samples - mean  # Shape: (M, D)
        
        # 2. Compute the Dual Covariance (Gram) Matrix. 
        # Shape: (M, M). For M=6, this is a tiny 6x6 matrix!
        # This is mathematically equivalent to computing the DxD covariance matrix.
        dual_cov = (centered @ centered.T) / M
        
        # 3. Get the eigenvalues. 
        eigenvalues = np.linalg.eigvalsh(dual_cov)
        
        # 4. A centered M x M matrix has exactly M-1 non-zero eigenvalues.
        # Extract the top active eigenvalues and clip to prevent floating point noise.
        active_eigenvalues = np.sort(eigenvalues)[-active_dims:]
        active_eigenvalues = np.clip(active_eigenvalues, 0.0, None)
        
        # 5. Add observation noise to the active dimensions
        log_det_active = np.sum(np.log(active_eigenvalues + sigma_squared))
        
        # 6. For the remaining (D - active_dims) empty dimensions, 
        # the eigenvalue is exactly 0. So adding noise just makes them sigma_squared.
        log_det_inactive = (D - active_dims) * np.log(sigma_squared)
        
        # 7. Total Log Determinant is the sum of active and inactive dimensions
        total_log_det = log_det_active + log_det_inactive
        
        entropy[i] = 0.5 * total_log_det + constant_term

    if len(mu_array.shape) == 2:
        return entropy[0]
    return entropy

def gaussian_entropy(mu_array: np.ndarray, sigma_squared: float) -> np.ndarray:
    """
    Calculate the entropy of multivariate Gaussian distributions with covariance
    Diag(1/M * Σ(μₘ²) - μ̄²) + σ²I in batch mode.
    """
    # Keep computations in at least float32 to avoid precision/compatibility issues.
    mu_array = np.asarray(mu_array, dtype=np.float32)

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

# 1. Native Resolution Preprocessing (No Resizing!)
def vgg16_preprocess(image: Image.Image) -> torch.Tensor:
    # We keep the pure 128x128 resolution
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    # Standard ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image - mean) / std

def dinov2_preprocess(image: Image.Image) -> torch.Tensor:
    # Resize directly to the ViT standard without cropping away the edges
    image = image.resize((224, 224), resample=Image.BICUBIC)

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
    
    # NEW: SigLIP (ViT-B/16 tuned on WebLI)
    if encoder_name == "siglip":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
        model.eval().to(device)

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            features = model.encode_image(image_batch)
            # OpenCLIP models sometimes don't L2 normalize by default like OpenAI CLIP does
            features = F.normalize(features, p=2, dim=1)
            return features

        return preprocess, encode_batch

    # NEW: OpenCLIP ViT-H/14 (Massive model tuned on LAION-2B)
    if encoder_name == "openclip_h14":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        model.eval().to(device)

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            features = model.encode_image(image_batch)
            features = F.normalize(features, p=2, dim=1)
            return features

        return preprocess, encode_batch

    if encoder_name == "dinov2_vits14_reg":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        model.eval().to(device)
        preprocess = dinov2_preprocess

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            features = model.forward_features(image_batch)
            # FIX 1: Return the global CLS token instead of patch tokens
            # This yields a (Batch, 384) tensor
            return features["x_norm_clstoken"]

        return preprocess, encode_batch

    
    if encoder_name == "vgg16":
        # Load the full model, including the classifier head
        full_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).eval().to(device)
        
        # We want to extract 'fc7' (the second fully connected layer before the final class predictions)
        # In torchvision's VGG16, the classifier is a Sequential block.
        # Indices: 0=fc6, 1=ReLU, 2=Dropout, 3=fc7, 4=ReLU, 5=Dropout, 6=fc8(logits)
        # We will use the output of fc7 + ReLU (up to index 4)
        feature_extractor = full_vgg.features
        avgpool = full_vgg.avgpool
        classifier = full_vgg.classifier[:5] 
        
        preprocess = vgg16_preprocess

        def encode_batch(image_batch: torch.Tensor) -> torch.Tensor:
            # 1. Pass through convolutional layers
            x = feature_extractor(image_batch)
            
            # 2. Pool to 7x7 (standard VGG behavior)
            x = avgpool(x)
            
            # 3. Flatten for fully connected layers
            x = torch.flatten(x, 1)
            
            # 4. Pass through fc6 and fc7 to get the 4096-D semantic vector
            features = classifier(x)
            
            # L2 Normalize to tame the variance and make it behave slightly more like CLIP
            features = F.normalize(features, p=2, dim=1)
            return features

        return preprocess, encode_batch

    raise ValueError(f"Unknown encoder: {encoder_name}")


def compute_generative_uncertainty(path, M, eu_type="entropy", encoder_name="clip", chunk_size=256):
    print(f"Loading samples from {M} models from {path} using encoder '{encoder_name}'")

    #### 1) Compute image features

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU-only mode requires CUDA.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    preprocess, encode_batch = build_encoder(encoder_name, device)

    # count number of images in path
    N = len(os.listdir(f"{path}/{0}/imgs"))
    feature_filename = f"{encoder_name}_features.npy"

    features_available = True 

    if not features_available:
        for m in tqdm(range(M), desc="Extracting features from models", unit="model"):
            print(f"Processing model {m}")
            image_vecs = []
            for i in range(N):
                image = Image.open(f"{path}/{m}/imgs/{i:05d}.png").convert("RGB")
                image = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    # OpenAI CLIP on CUDA often returns float16; store as float32 for NumPy linalg.
                    image_vecs.append(encode_batch(image).float().cpu())

            image_vecs = torch.concat(image_vecs, dim=0)
            print(image_vecs.shape)
            features_path = Path(path) / str(m) / feature_filename
            features_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(features_path, image_vecs.numpy())
    else:
        print("features were already computed")

    #### 2) Compute the entropy of the semantic likelihood

    feature_paths = [Path(path) / str(m) / feature_filename for m in range(M)]
    first_feature = np.load(feature_paths[0], mmap_mode="r")

    if first_feature.ndim == 2:
        # GLOBAL ENCODER (e.g., CLIP), saved shape per model: (N, D)
        N, _ = first_feature.shape
        eu = np.empty(N, dtype=np.float64)
        print(f"Loaded GLOBAL feature shape per model: {tuple(first_feature.shape)}")

        if eu_type != "entropy":
            raise ValueError(f"Unknown epistemic uncertainty type: {eu_type}")

        for start in tqdm(range(0, N, chunk_size), desc="Computing uncertainty", unit="chunk"):
            end = min(start + chunk_size, N)
            chunk_features = []
            for feature_path in feature_paths:
                feature_m = np.load(feature_path, mmap_mode="r")
                chunk_features.append(np.asarray(feature_m[start:end]))

            # (M, B, D) -> (B, M, D)
            features_chunk = np.transpose(np.stack(chunk_features, axis=0), (1, 0, 2))
            eu[start:end] = exact_gaussian_entropy(features_chunk, sigma_squared=1e-3)

    elif first_feature.ndim == 3:
        # DENSE ENCODER (e.g., VGG-16, DINOv2), saved shape per model: (N, P, D)
        N, P, _ = first_feature.shape
        eu = np.empty(N, dtype=np.float64)
        print(f"Loaded DENSE feature shape per model: {tuple(first_feature.shape)}")

        if eu_type != "entropy":
            raise ValueError(f"Unknown epistemic uncertainty type: {eu_type}")

        for start in tqdm(range(0, N, chunk_size), desc="Computing uncertainty", unit="chunk"):
            end = min(start + chunk_size, N)
            chunk_features = []
            for feature_path in feature_paths:
                feature_m = np.load(feature_path, mmap_mode="r")
                chunk_features.append(np.asarray(feature_m[start:end]))

            # (M, B, P, D) -> (B, M, P, D)
            features_chunk = np.transpose(np.stack(chunk_features, axis=0), (1, 0, 2, 3))
            B = features_chunk.shape[0]

            # Same math as before: entropy per patch, then max over patches.
            features_reshaped = features_chunk.reshape(B * P, M, features_chunk.shape[-1])
            patch_eu = gaussian_entropy(features_reshaped, sigma_squared=1e-4)
            eu[start:end] = patch_eu.reshape(B, P).max(axis=1)

    else:
        raise ValueError(f"Unexpected feature dimensions: {first_feature.ndim}")

    print(f"Saving: {path}/{eu_type}_{encoder_name}.npy")
    eu_path = Path(path) / f"{eu_type}_{encoder_name}.npy"
    eu_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(eu_path, eu)

if __name__ == "__main__":
    args = parse_args()
    compute_generative_uncertainty(args.path, args.M, encoder_name=args.encoder, chunk_size=args.chunk_size)
