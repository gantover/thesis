import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

# Import existing utilities from your project
from .lora_ensemble import inject_lora
from .laplace_hessian import compute_hessian_approx, _get_diffusion_target
from .ensemble_weights import (
    LaplaceDecoderAdapter, 
    _sample_from_posterior, 
    sample_subset_model,
)

from .utils import get_param_str_la_lora
from .model_loading import load_base_model
from .config import LaplaceLoraEnsembleConfig, DeepEnsembleConfig

def get_lora_indices(model):
    """
    Extracts the flat parameter indices for all LoRA parameters in the model.
    This ensures the Hessian is only computed over the low-rank adapters.
    """
    indices = []
    offset = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        # Only target the LoRA matrices
        if "lora_A" in name or "lora_B" in name:
            indices.extend(range(offset, offset + numel))
        offset += numel
        
    if not indices:
        raise ValueError("No LoRA parameters found. Did you forget to inject_lora()?")
        
    return torch.tensor(indices, dtype=torch.long)


def train_lora_map(
    base_model, 
    diffusion, 
    real_data, 
    epochs=10, 
    batch_size=2048, 
    lr=1e-3, 
    r=4, 
    alpha=1.0, 
    device="cpu"
):
    """
    Fine-tunes the base model using LoRA to obtain the MAP estimate.
    """
    print("Training MAP estimate for LoRA parameters...")
    map_model = copy.deepcopy(base_model)
    inject_lora(map_model, r=r, alpha=alpha)
    map_model.to(device)
    map_model.train()
    
    # Optimize ONLY the parameters that require grad (lora_A and lora_B)
    optimizer = torch.optim.Adam([p for p in map_model.parameters() if p.requires_grad], lr=lr)
    dataset = torch.utils.data.TensorDataset(torch.as_tensor(real_data, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for (x_0,) in loader:
            x_0 = x_0.to(device)
            
            t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=device)
            noise = torch.randn_like(x_0)
            x_t = diffusion.q_sample(x_0, t, noise=noise)
            target = _get_diffusion_target(diffusion, x_0, x_t, noise, t)
            
            optimizer.zero_grad()
            out = map_model(x_t, t) 
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % max(1, (epochs // 5)) == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:02d}/{epochs} | MAP Loss: {total_loss/len(loader):.4f}")
            
    map_model.eval()
    return map_model


def build_laplace_lora_ensemble(
    la: LaplaceLoraEnsembleConfig,
    de: DeepEnsembleConfig,
    device: torch.device,
    diffusion
):
    """
    End-to-end pipeline for Laplace LoRA:
    1. Load base model.
    2. Train MAP LoRA estimate.
    3. Compute Hessian over LoRA parameters.
    4. Sample LoRA parameters to form an ensemble.
    """
    print(f"Building Laplace LoRA ensemble: curvature={la.curvature}, approximation={la.approximation}, r={la.r}")

    # 1. Load Base Model and Data
    chkpt_dir = Path(de.trained_models_dir.format(seed=0))
    base_model = load_base_model(de_config=de, device=device)

    real_dataset_path = chkpt_dir / "real_dataset.npy"
    if not os.path.exists(real_dataset_path):
        raise FileNotFoundError(f"Missing calibration data: {real_dataset_path}")
    real_data = np.load(real_dataset_path)

    if la.weight_sampling_seed is not None:
        torch.manual_seed(la.weight_sampling_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(la.weight_sampling_seed)

    la_chkpt_dir = Path(la.la_lora_sampled_models_dir)
    la_chkpt_dir.mkdir(parents=True, exist_ok=True)

    # 2. Train MAP LoRA estimate
    map_model = train_lora_map(
        base_model, diffusion, real_data, 
        epochs=la.map_epochs, batch_size=la.laplace_batch_size, lr=la.map_lr, r=la.r, alpha=la.alpha, device=device
    )

    # 3. Setup Laplace over LoRA parameters
    adapter = LaplaceDecoderAdapter(copy.deepcopy(map_model)).to(device)
    flat_indices = get_lora_indices(map_model).to(device)
    m_eff = len(flat_indices)
    print(f"Total LoRA parameters for Laplace approximation: {m_eff}")

    # 4. Compute Hessian Approximation
    sigma = compute_hessian_approx(
        adapter=adapter,
        diffusion=diffusion,
        real_data=real_data,
        curvature=la.curvature,
        flat_indices=flat_indices,
        n_batches=la.laplace_batches,
        batch_size=la.laplace_batch_size,
        prior_precision=la.prior_precision,
        device=device,
        approximation=la.approximation,
    )

    # Save initial covariance stats
    param_str = get_param_str_la_lora(lora_config=la)
    np.save(la_chkpt_dir / f"lora_pre_sigma_{param_str}.npy", sigma.cpu().numpy())
    
    diag = torch.diag(sigma) if sigma.ndim == 2 else sigma
    print(f"LoRA posterior sigma stats ({la.approximation}): min={diag.min().item():.3e}, median={diag.median().item():.3e}, max={diag.max().item():.3e}")

    # 5. Sample from the Posterior
    mean_subset = parameters_to_vector(map_model.parameters()).detach()[flat_indices]
    sampled_layers, sigma_scaled = _sample_from_posterior(
        sigma, mean_subset, de.M, la.approximation, la.temperature, m_eff, device
    )
    np.save(la_chkpt_dir / f"lora_post_sigma_{param_str}.npy", sigma_scaled.cpu().numpy())

    # 6. Build and save the ensemble models
    models = [map_model] # Include the MAP estimate as the 0-th member
    
    for i in range(de.M):
        sampled_model = sample_subset_model(
            base_model=map_model,
            sampled_vector=sampled_layers[i],
            flat_indices=flat_indices,
            device=device,
        )
        model_cache_path = la_chkpt_dir / f"la_lora_sample_{i}_{param_str}.pt"
        torch.save(sampled_model.state_dict(), model_cache_path)
        print(f"Saved Laplace LoRA sampled model {i+1}/{de.M} to: {model_cache_path}")
        models.append(sampled_model)

    return models