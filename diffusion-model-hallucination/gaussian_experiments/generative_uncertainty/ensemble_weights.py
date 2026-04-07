import copy
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ddpm_torch.toy import Decoder, GaussianDiffusion, get_beta_schedule
from .model_loading import load_base_model
from .utils import get_param_str


class LaplaceCalibrationDataset(torch.utils.data.Dataset):
    """Dataset that samples calibration points with replacement."""
    def __init__(self, real_data, total_samples):
        super().__init__()
        if total_samples <= 0:
            raise ValueError("total_samples must be > 0")

        self.data = torch.as_tensor(real_data, dtype=torch.float32)
        if self.data.ndim != 2:
            raise ValueError("Expected real_data shape (N, D)")

        self.total_samples = total_samples
        self.n = self.data.shape[0]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        _ = idx
        sample_idx = torch.randint(0, self.n, (1,), dtype=torch.long).item()
        return self.data[sample_idx]

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

def _ef_diag_batch(adapter, x_t, y, t, flat_indices, params_dict, buffers_dict):
    """Per-sample EF squared gradients at selected flat_indices. Returns (B, m) tensor."""
    def loss_single(x, y, t, params_dict, buffers_dict):
        x, y, t = x.unsqueeze(0), y.unsqueeze(0), t.unsqueeze(0)
        output = torch.func.functional_call(adapter, (params_dict, buffers_dict), (x, t))
        loss = 0.5 * torch.nn.functional.mse_loss(output, y, reduction="sum")
        return loss

    grad_fn = torch.func.grad(loss_single, argnums=3)
    batch_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, 0, None, None))
    batch_grad = batch_grad_fn(x_t, y, t, params_dict, buffers_dict)
    Gs = torch.cat([g.flatten(start_dim=1) for g in batch_grad.values()], dim=1)
    if flat_indices is not None:
        return Gs[:, flat_indices]
    return Gs

def _ggn_diag_batch(adapter, x_t, t, flat_indices, d, params_dict, buffers_dict):
    """Per-sample GGN squared Jacobians at selected flat_indices. Returns (B*d, m) tensor."""
    Gs_list = []
    for k in range(d):
        def _output_k(x, t, params_dict, buffers_dict, _k=k):
            x, t = x.unsqueeze(0), t.unsqueeze(0)
            return torch.func.functional_call(adapter, (params_dict, buffers_dict), (x, t))[0, _k]

        grad_fn = torch.func.grad(_output_k, argnums=2)
        batch_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, None, None))
        batch_grad_k = batch_grad_fn(x_t, t, params_dict, buffers_dict)
        flat_k = torch.cat([g.flatten(start_dim=1) for g in batch_grad_k.values()], dim=1)
        Gs_list.append(flat_k)

    Gs = torch.cat(Gs_list, dim=0)
    if flat_indices is not None:
        return Gs[:, flat_indices]
    return Gs

def _get_diffusion_target(diffusion, x_0, x_t, noise, t):
    if diffusion.model_mean_type == "eps":
        return noise
    if diffusion.model_mean_type == "x_0":
        return x_0
    return diffusion.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)[0]

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

def compute_hessian_approx(adapter, diffusion, real_data, curvature, flat_indices, n_batches, batch_size, prior_precision, device, approximation="diagonal"):
    if curvature not in {"ef", "ggn"}:
        raise ValueError(f"Unsupported curvature '{curvature}'. Use 'ef' or 'ggn'.")
    if approximation not in {"diagonal", "full", "kfac", "icla"}:
        raise ValueError(f"Unsupported approximation '{approximation}'. Use 'diagonal', 'full', 'kfac', or 'icla'.")
    flat_indices = flat_indices.to(device=device, dtype=torch.long)
    for p in adapter.parameters():
        p.requires_grad_(True)
    dataset = LaplaceCalibrationDataset(real_data=real_data, total_samples=n_batches * batch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    params_dict = dict(adapter.named_parameters())
    buffers_dict = dict(adapter.named_buffers())
    d = real_data.shape[-1]
    m = len(flat_indices)

    if approximation in {"full", "kfac"}:
        H = torch.zeros((m, m), device=device)
    elif approximation == "diagonal": # diagonal or icla
        H_diag = torch.zeros(m, device=device)

    if approximation == "kfac":
        target_modules = []
        for name, mod in adapter.named_modules():
            if hasattr(mod, "weight") and hasattr(mod, "in_features") and hasattr(mod, "out_features"):
                # Make sure we only grab the one that matches our selected subset/last_layer_name exactly
                # Here we assume the target is adapter.decoder.<last_layer_name>
                target_modules.append((name, mod))
        
        if not target_modules:
            raise ValueError("KFAC failed to find linear module.")
            
        target_module = None
        for name, mod in target_modules:
            # Match strictly against the expected subset layer
            if hasattr(adapter, "decoder") and hasattr(adapter.decoder, "out_fc"):
                if mod is getattr(adapter.decoder, "out_fc"):
                    target_module = mod
                    break
        
        if target_module is None:
            target_module = target_modules[-1][1]  # Fallback
        
        d_in = target_module.in_features
        d_out = target_module.out_features
        in_feat_sz = d_in + (1 if target_module.bias is not None else 0)
        
        if in_feat_sz * d_out != m:
            raise ValueError(f"KFAC size mismatch: module has {in_feat_sz * d_out} params but m={m}")
            
        A_sum = torch.zeros((in_feat_sz, in_feat_sz), device=device)
        B_sum = torch.zeros((d_out, d_out), device=device)
        
        a_cache = []
        g_cache = []
        def fw_hook(mod, x_in, y_out):
            a = x_in[0].detach()
            if mod.bias is not None:
                a = torch.cat([a, torch.ones(a.shape[0], 1, device=a.device)], dim=1)
            a_cache.append(a)
        def bw_hook(mod, g_in, g_out):
            g_cache.append(g_out[0].detach())
            
        h_fw = target_module.register_forward_hook(fw_hook)
        h_bw = target_module.register_full_backward_hook(bw_hook)
        n_total = 0

    hessian_factor = 0.5
    for x_0 in loader:
        x_0 = x_0.to(device)
        t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=device)
        noise = torch.randn_like(x_0)
        x_t = diffusion.q_sample(x_0, t, noise=noise)
        
        if approximation == "kfac":
            a_cache.clear()
            g_cache.clear()
            adapter.zero_grad()
            y_pred = adapter(x_t, t)
            
            if curvature == "ggn":
                a = a_cache[0].clone()
                # GGN requires taking the gradient of each output dimension separately
                # wrt to the pre-activation
                for k in range(y_pred.shape[1]):
                    adapter.zero_grad()
                    g_cache.clear()
                    # The sum over batch matches vmap logic handling independent elements
                    # Multiplying by sqrt(2) matches PyTorch's MSE loss reduction="sum" 
                    # 2nd derivative being 2 on the diagonal. => g * sqrt(2) -> g^2 * 2
                    y_pred[:, k].sum().backward(retain_graph=True)
                    g = g_cache[-1]
                    B_sum += torch.einsum("bi,bj->ij", g, g)
                A_sum += torch.einsum("bi,bj->ij", a, a)
            else:
                y = _get_diffusion_target(diffusion=diffusion, x_0=x_0, x_t=x_t, noise=noise, t=t)
                loss = 0.5 * torch.nn.functional.mse_loss(y_pred, y, reduction="sum")
                loss.backward()
                a = a_cache[0]
                g = g_cache[-1]
                A_sum += torch.einsum("bi,bj->ij", a, a)
                B_sum += torch.einsum("bi,bj->ij", g, g)
            
            n_total += x_0.shape[0]
        else:
            if curvature == "ggn":
                Gs = _ggn_diag_batch(adapter, x_t, t, flat_indices, d, params_dict, buffers_dict)
            else:
                y = _get_diffusion_target(diffusion=diffusion, x_0=x_0, x_t=x_t, noise=noise, t=t)
                Gs = _ef_diag_batch(adapter, x_t, y, t, flat_indices, params_dict, buffers_dict)
            
            if approximation == "full":
                H += hessian_factor * torch.einsum("bp,bq->pq", Gs.detach(), Gs.detach())
            elif approximation == "diagonal":
                H_diag += hessian_factor * torch.einsum("bp,bp->p", Gs.detach(), Gs.detach())

    if approximation == "kfac":
        h_fw.remove()
        h_bw.remove()
        H_kfac = hessian_factor * torch.kron(B_sum, A_sum) / max(n_total, 1)
        
        idx_kfac_to_pt = torch.zeros(m, dtype=torch.long, device=device)
        for o in range(d_out):
            for c in range(d_in):
                idx_kfac_to_pt[o * in_feat_sz + c] = o * d_in + c
            if target_module.bias is not None:
                idx_kfac_to_pt[o * in_feat_sz + d_in] = d_out * d_in + o
                
        H[idx_kfac_to_pt.unsqueeze(1), idx_kfac_to_pt.unsqueeze(0)] = H_kfac

    if approximation in {"full", "kfac"}:
        sigma = torch.linalg.inv(H + prior_precision * torch.eye(m, device=device))
    if approximation == "diagonal":
        sigma = 1.0 / (H_diag + prior_precision)
    if approximation == "icla":
        # ICLA : Identity Curvature Laplace Approximation
        sigma = 1.0 / (prior_precision * torch.ones(m, device=device))  # Start with identity approximation
        # # ICLA: iterative correction of diagonal approximation with low-rank EF info
        # # See "Iteratively Corrected Laplace Approximation" (ICLA) method in https://arxiv.org/abs/2211.13227
        # sigma_diag = 1.0 / (H_diag + prior_precision)
        # sigma_icla = sigma_diag.clone()
        # n_icla_iters = 5
        # for _ in range(n_icla_iters):
        #     H_icla = hessian_factor * torch.einsum("bp,bq->pq", Gs.detach(), Gs.detach() * sigma_icla[flat_indices].unsqueeze(0))
        #     sigma_icla = 1.0 / (H_diag + H_icla.diag() + prior_precision)
        # sigma = sigma_icla

    return sigma

def sample_subset_model(base_model, sampled_vector, flat_indices, device):
    sampled = copy.deepcopy(base_model)
    flat = parameters_to_vector(sampled.parameters()).detach().clone()
    flat[flat_indices.to(flat.device)] = sampled_vector.to(flat.device)
    vector_to_parameters(flat, sampled.parameters())
    sampled.to(device)
    sampled.eval()
    return sampled

def build_laplace_ensemble(
    trained_models_dir,
    la_sampled_models_dir,
    diffusion,
    device,
    sel_generation=0,
    M=5,
    laplace_batches=64,
    laplace_batch_size=2048,
    prior_precision=1e-2,
    sample_temperature=1.0,
    weight_sampling_seed=None,
    last_layer_name="out_fc",
    subset="last_layer",
    curvature="ef",
    approximation="diagonal",
    m=1000,
    subset_seed=42,
):
    print(f"Building Laplace ensemble: subset={subset}, curvature={curvature}, approximation={approximation}")

    chkpt_dir = Path(trained_models_dir.format(seed=0))
    base_model = load_base_model(trained_models_dir=trained_models_dir, sel_generation=sel_generation, device=device)

    real_dataset_path = chkpt_dir / "real_dataset.npy"
    if not os.path.exists(real_dataset_path):
        raise FileNotFoundError(f"Missing calibration data: {real_dataset_path}")
    real_data = np.load(real_dataset_path)

    if weight_sampling_seed is not None:
        torch.manual_seed(weight_sampling_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(weight_sampling_seed)

    la_chkpt_dir = Path(la_sampled_models_dir)
    la_chkpt_dir.mkdir(parents=True, exist_ok=True)

    adapter = LaplaceDecoderAdapter(copy.deepcopy(base_model)).to(device)

    flat_indices = get_subnetwork_indices(base_model, subset, m, subset_seed, last_layer_name).to(device)
    m_eff = len(flat_indices)

    sigma = compute_hessian_approx(
        adapter=adapter,
        diffusion=diffusion,
        real_data=real_data,
        curvature=curvature,
        flat_indices=flat_indices,
        n_batches=laplace_batches,
        batch_size=laplace_batch_size,
        prior_precision=prior_precision,
        device=device,
        approximation=approximation,
    )

    np.save(la_chkpt_dir / "pre_sigma.npy", sigma.cpu().numpy())
    
    mean_subset = parameters_to_vector(base_model.parameters()).detach()[flat_indices]

    if approximation in {"full", "kfac"}:
        sigma = sigma * (sample_temperature ** 2)
        
        # Ensure exact symmetry before Cholesky/MultivariateNormal
        sigma = (sigma + sigma.transpose(-2, -1)) / 2.0
        
        print(
            f"{subset.capitalize()} subset posterior sigma stats ({approximation}): "
            f"min={torch.diag(sigma).min().item():.3e}, median={torch.diag(sigma).median().item():.3e}, max={torch.diag(sigma).max().item():.3e}"
        )
        
        dist = torch.distributions.MultivariateNormal(mean_subset, covariance_matrix=sigma + 1e-6 * torch.eye(m_eff, device=device))
        sampled_layers = dist.rsample((M,))
    else:
        sigma = torch.clamp(sigma, min=1e-12) * (sample_temperature ** 2)
        print(
            f"{subset.capitalize()} subset posterior sigma stats ({approximation}): "
            f"min={sigma.min().item():.3e}, median={sigma.median().item():.3e}, max={sigma.max().item():.3e}"
        )
        
        eps = torch.randn(M, m_eff, device=device)
        sampled_layers = mean_subset.unsqueeze(0) + eps * sigma.sqrt().unsqueeze(0)

    np.save(la_chkpt_dir / "post_sigma.npy", sigma.cpu().numpy())

    models = [base_model]
    param_str = get_param_str(prior_precision, approximation, curvature, subset, m=m, temperature=sample_temperature)

    for i in range(M):
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
