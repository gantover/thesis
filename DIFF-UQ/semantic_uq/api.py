import os
import glob
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image
from typing import List, Optional

from .encoders import build_encoder
from .entropy import exact_gaussian_entropy, gaussian_entropy, trace_variance, distance_to_anchor
from .perturbations import ImagePerturber

def compute_uncertainty_precomputed(
    path: str,
    m_samples: int,
    encoder_name: str = "clip",
    entropy_calculation: str = "diagonal",
    anchor_base: bool = True,
    chunk_size: int = 256,
    eu_type: str = "entropy",
    device: str = "cuda"
) -> np.ndarray:
    """
    Computes uncertainty from pre-computed images saved in M directories under `path`.
    Example directory structure:
       path/0/imgs/00000.png
       path/0/imgs/00001.png
       ...
       path/M-1/imgs/00000.png
    """
    dev = torch.device(device)
    preprocess, encode_batch, preprocess_gpu = build_encoder(encoder_name, dev)

    # 1) Compute image features or load them if they exist
    # Determine N based on how many images in the `0` folder
    img_dir_0 = Path(path) / "0" / "imgs"
    if not img_dir_0.exists():
        raise FileNotFoundError(f"Missing base directory {img_dir_0}")
        
    num_images = len(os.listdir(img_dir_0))
    feature_filename = f"{encoder_name}_features.npy"
    
    for m in tqdm(range(m_samples), desc="Extracting features from models"):
        features_path = Path(path) / str(m) / feature_filename
        if features_path.exists():
            continue  # Already computed
            
        image_vecs = []
        for i in range(num_images):
            img_path = Path(path) / str(m) / "imgs" / f"{i:05d}.png"
            image = Image.open(img_path).convert("RGB")
            processed = preprocess(image).unsqueeze(0).to(dev)

            with torch.no_grad():
                image_vecs.append(encode_batch(processed).float().cpu())

        image_vecs = torch.concat(image_vecs, dim=0)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(features_path, image_vecs.numpy())

    # 2) Compute the entropy
    feature_paths = [Path(path) / str(m) / feature_filename for m in range(m_samples)]
    first_feature = np.load(feature_paths[0], mmap_mode="r")
    
    eu = np.empty(num_images, dtype=np.float64)

    for start in tqdm(range(0, num_images, chunk_size), desc="Computing uncertainty"):
        end = min(start + chunk_size, num_images)
        chunk_features = []
        for fp in feature_paths:
            chunk_features.append(np.asarray(np.load(fp, mmap_mode="r")[start:end]))

        # Format: (Batch, M, D)
        features_chunk = np.transpose(np.stack(chunk_features, axis=0), (1, 0, 2))
        
        # Compute entropy using sigma_squared=1e-8. This prevents completely swallowing
        # the small variances of L2-normalized encoders (like SigLIP) while maintaining a noise floor.
        if entropy_calculation == "full":
            eu[start:end] = exact_gaussian_entropy(features_chunk, sigma_squared=1e-8, anchor_base=anchor_base)
        elif entropy_calculation == "diagonal":
            eu[start:end] = gaussian_entropy(features_chunk, sigma_squared=1e-8, anchor_base=anchor_base)
        elif entropy_calculation == "trace":
            eu[start:end] = trace_variance(features_chunk, anchor_base=anchor_base)
        elif entropy_calculation == "distance":
            eu[start:end] = distance_to_anchor(features_chunk)
        else:
            raise ValueError(f"Unknown entropy calculation mode: {entropy_calculation}")

    eu_path = Path(path) / f"{eu_type}_{encoder_name}.npy"
    eu_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(eu_path, eu)
    return eu


def compute_uncertainty_onthefly(
    base_dir: str,
    m_samples: int = 10,
    noise_scale: float = 0.02,
    use_transforms: bool = False,
    encoder_name: str = "clip",
    entropy_calculation: str = "diagonal",
    anchor_base: bool = True,
    batch_size: int = 32,
    device: str = "cuda"
) -> np.ndarray:
    """
    Computes uncertainty dynamically from a single directory of base images,
    applying noise/transforms in-memory without saving intermediate files.
    """
    dev = torch.device(device)
    import torchvision.transforms.functional as TF
    preprocess, encode_batch, preprocess_gpu = build_encoder(encoder_name, dev)
    perturber = ImagePerturber(m_samples=m_samples, noise_scale=noise_scale, use_transforms=use_transforms)
    
    image_paths = sorted(glob.glob(os.path.join(base_dir, "*.png")) + glob.glob(os.path.join(base_dir, "*.jpg")))
    if not image_paths:
        raise ValueError(f"No images found in {base_dir}")

    N = len(image_paths)
    eu = np.zeros(N, dtype=np.float64)

    print(f"Applying on-the-fly UQ mapping over {N} images (M={m_samples})")

    # In on-the-fly, M images are generated *per base image*.
    use_fast_gpu_path = not use_transforms

    for start in tqdm(range(0, N, batch_size), desc="Computing on-the-fly uncertainty"):
        end = min(start + batch_size, N)
        current_batch_paths = image_paths[start:end]
        
        if use_fast_gpu_path:
            # 1. Read base images to batched PyTorch tensor
            all_raw = []
            for path in current_batch_paths:
                img = Image.open(path).convert('RGB')
                all_raw.append(TF.to_tensor(img))
            
            base_batch = torch.stack(all_raw).to(dev) # Shape: (B, 3, H, W)
            B, C, H, W = base_batch.shape
            
            # 2. Duplicate on GPU to get M copies per base image
            # Unqueeze to (B, 1, C, H, W) -> expand to (B, M, C, H, W)
            batch_tensors = base_batch.unsqueeze(1).expand(-1, m_samples, -1, -1, -1).contiguous()
            
            # 3. Add noise strictly to copies 1 through M-1 (Keep copy 0 clean)
            if noise_scale > 0 and m_samples > 1:
                noise = torch.randn_like(batch_tensors[:, 1:]) * noise_scale
                batch_tensors[:, 1:] = torch.clamp(batch_tensors[:, 1:] + noise, 0.0, 1.0)
                
            # 4. Collapse dimension for the encoder (B*M, C, H, W)
            batch_tensors = batch_tensors.view(B * m_samples, C, H, W)
            
            # 5. Apply the pure-GPU ImageNet normalize/resize pipeline
            batch_tensors = preprocess_gpu(batch_tensors)
            
        else:
            # Slower PIL fallback for spatial transforms (RandomResizedCrop)
            all_tensors = []
            for path in current_batch_paths:
                img = Image.open(path).convert('RGB')
                perturbed_pil_list = perturber(img)
                for p_img in perturbed_pil_list:
                    all_tensors.append(preprocess(p_img))
            batch_tensors = torch.stack(all_tensors).to(dev)
        
        # 6. Encode in memory-safe chunks to prevent GPU OOM (especially at M=512)
        chunk_lim = 128
        features_list = []
        with torch.no_grad():
            for c_start in range(0, batch_tensors.shape[0], chunk_lim):
                feat = encode_batch(batch_tensors[c_start:c_start + chunk_lim])
                features_list.append(feat)
                
        features = torch.cat(features_list, dim=0)
            
        # Reshape back to (B, M, D)
        features = features.view(len(current_batch_paths), m_samples, -1).cpu().numpy()
        
        # Compute entropy using sigma_squared=1e-8. This acts as a safe numerical
        # stabilizer and preserves tiny variances from L2-normalized embeddings.
        if entropy_calculation == "full":
            batch_eu = exact_gaussian_entropy(features, sigma_squared=1e-8, anchor_base=anchor_base)
        elif entropy_calculation == "diagonal":
            batch_eu = gaussian_entropy(features, sigma_squared=1e-8, anchor_base=anchor_base)
        elif entropy_calculation == "trace":
            batch_eu = trace_variance(features, anchor_base=anchor_base)
        elif entropy_calculation == "distance":
            batch_eu = distance_to_anchor(features)
        else:
            raise ValueError(f"Unknown entropy calculation mode: {entropy_calculation}")
            
        eu[start:end] = batch_eu
        
    eu_path = Path(base_dir).parent.parent / f"entropy_{encoder_name}.npy"
    eu_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(eu_path, eu)
    return eu
