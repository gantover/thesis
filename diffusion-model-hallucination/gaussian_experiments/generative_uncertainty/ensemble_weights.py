import copy
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ddpm_torch.toy import GaussianDiffusion, get_beta_schedule
from .laplace_hessian import compute_hessian_approx
from .model_loading import load_base_model
from .utils import get_param_str_la
from .config import LaplaceEnsembleConfig, DeepEnsembleConfig


class LaplaceDecoderAdapter(torch.nn.Module):
    """Wraps Decoder with a vmap-safe timestep embedding (avoids TorchScript inside functional_call)."""
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    @staticmethod
    def _timestep_embedding(timesteps, embed_dim, dtype=torch.float32):
        half_dim = embed_dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(-torch.arange(half_dim, dtype=dtype, device=timesteps.device) * scale)
        emb = torch.outer(timesteps.reshape(-1).to(dtype), freqs)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embed_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, [0, 1])
        return emb

    def forward(self, x_t, t):
        t = t.to(torch.long)
        t_emb = self._timestep_embedding(t, self.decoder.mid_features)
        t_emb = self.decoder.t_proj(t_emb)
        out = self.decoder.in_fc(x_t)
        out = self.decoder.temp_fc(out, t_emb=t_emb)
        return self.decoder.out_fc(self.decoder.out_norm(out))


def get_diffusion(timesteps=1000):
    betas = get_beta_schedule("linear", beta_start=0.001, beta_end=0.2, timesteps=timesteps)
    return GaussianDiffusion(
        betas=betas,
        model_mean_type="eps",
        model_var_type="fixed-large",
        loss_type="mse",
    )


def get_subnetwork_indices(model, subset, m, subset_seed, last_layer_name):
    total_params = sum(p.numel() for p in model.parameters())
    if subset == "last_layer":
        indices = []
        offset = 0
        for name, p in model.named_parameters():
            numel = p.numel()
            if last_layer_name in name:
                indices.extend(range(offset, offset + numel))
            offset += numel
        if not indices:
            raise ValueError(f"No parameters matched last_layer_name='{last_layer_name}'.")
        return torch.tensor(indices, dtype=torch.long)
    elif subset == "multi_layer":
        indices = []
        offset = 0
        if isinstance(last_layer_name, str):
            layer_names = [x.strip() for x in last_layer_name.split(',')]
        else:
            layer_names = last_layer_name

        for name, p in model.named_parameters():
            numel = p.numel()
            if any(l_name in name for l_name in layer_names):
                indices.extend(range(offset, offset + numel))
            offset += numel
        if not indices:
            raise ValueError(f"No parameters matched multi_layer names: {layer_names}")
        return torch.tensor(indices, dtype=torch.long)
    elif subset == "random":
        m_eff = min(m, total_params)
        if m_eff < m:
            print(f"Warning: requested m={m} exceeds total_params={total_params}; using m={m_eff}.")
        rng = torch.Generator().manual_seed(subset_seed)
        return torch.randperm(total_params, generator=rng)[:m_eff]
    else:
        raise ValueError(f"Unknown subset mode: {subset}")

# Posterior sampling helpers

def _sample_from_posterior(sigma, mean, M, approximation, temperature, m_eff, device):
    """Draw M weight samples from the Laplace posterior.

    Returns ``(sampled_layers, sigma_scaled)`` where ``sampled_layers`` is a
    ``(M, m_eff)`` tensor and ``sigma_scaled`` is the temperature-adjusted
    covariance (for saving to disk).
    """
    if approximation in {"full", "kfac"}:
        sigma = (sigma + sigma.T) / 2 * temperature ** 2
        dist = torch.distributions.MultivariateNormal(
            mean, covariance_matrix=sigma + 1e-6 * torch.eye(m_eff, device=device)
        )
        return dist.rsample((M,)), sigma
    elif approximation in {"diagonal"}:
        sigma = torch.clamp(sigma, min=1e-12) * temperature ** 2
        eps = torch.randn(M, m_eff, device=device)
        return mean.unsqueeze(0) + eps * sigma.sqrt().unsqueeze(0), sigma
    else:
        raise ValueError(f"Unsupported approximation '{approximation}'. Expected 'full', 'kfac', 'diagonal'")


def _log_sigma_stats(sigma, approximation, subset):
    diag = torch.diag(sigma) if sigma.ndim == 2 else sigma
    print(
        f"{subset.capitalize()} subset posterior sigma stats ({approximation}): "
        f"min={diag.min().item():.3e}, median={diag.median().item():.3e}, max={diag.max().item():.3e}"
    )

# Public API

def sample_subset_model(base_model, sampled_vector, flat_indices, device):
    sampled = copy.deepcopy(base_model)
    flat = parameters_to_vector(sampled.parameters()).detach().clone()
    flat[flat_indices.to(flat.device)] = sampled_vector.to(flat.device)
    vector_to_parameters(flat, sampled.parameters())
    sampled.to(device)
    sampled.eval()
    return sampled


def build_laplace_ensemble(
    de: DeepEnsembleConfig,
    la: LaplaceEnsembleConfig,
    device: torch.device,
    diffusion
):
    print(f"Building Laplace ensemble: subset={la.subset}, curvature={la.curvature}, approximation={la.approximation}")

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

    la_chkpt_dir = Path(la.la_sampled_models_dir)
    la_chkpt_dir.mkdir(parents=True, exist_ok=True)

    adapter = LaplaceDecoderAdapter(copy.deepcopy(base_model)).to(device)
    flat_indices = get_subnetwork_indices(base_model, la.subset, la.m, la.subset_seed, la.last_layer_name).to(device)
    m_eff = len(flat_indices)

    sigma, H = compute_hessian_approx(
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

    param_str = get_param_str_la(la_config=la)
    
    # Extract diagonal Fisher
    if la.approximation in {"full", "kfac"}:
        H_diag = torch.diag(H)
    else:
        H_diag = H
    np.save(la_chkpt_dir / f"fisher_diag_{param_str}.npy", H_diag.cpu().numpy())

    np.save(la_chkpt_dir / f"pre_sigma_{param_str}.npy", sigma.cpu().numpy())
    _log_sigma_stats(sigma, la.approximation, la.subset)

    mean_subset = parameters_to_vector(base_model.parameters()).detach()[flat_indices]
    sampled_layers, sigma_scaled = _sample_from_posterior(
        sigma, mean_subset, de.M, la.approximation, la.temperature, m_eff, device
    )
    np.save(la_chkpt_dir / f"post_sigma_{param_str}.npy", sigma_scaled.cpu().numpy())

    models = [base_model]

    for i in range(de.M):
        sampled_model = sample_subset_model(
            base_model=base_model,
            sampled_vector=sampled_layers[i],
            flat_indices=flat_indices,
            device=device,
        )
        model_cache_path = la_chkpt_dir / f"la_sample_{i}_{param_str}.pt"
        torch.save(sampled_model.state_dict(), model_cache_path)
        print(f"Saved sampled model: {model_cache_path}")
        models.append(sampled_model)

    return models
