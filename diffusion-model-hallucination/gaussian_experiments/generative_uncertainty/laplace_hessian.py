import torch


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


def _get_diffusion_target(diffusion, x_0, x_t, noise, t):
    if diffusion.model_mean_type == "eps":
        return noise
    if diffusion.model_mean_type == "x_0":
        return x_0
    return diffusion.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)[0]


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


def _find_kfac_target_module(adapter):
    """Return the linear module to apply KFAC to (prefers adapter.decoder.out_fc)."""
    linear_modules = [
        (name, mod) for name, mod in adapter.named_modules()
        if hasattr(mod, "weight") and hasattr(mod, "in_features") and hasattr(mod, "out_features")
    ]
    if not linear_modules:
        raise ValueError("KFAC failed to find any linear module.")
    if hasattr(adapter, "decoder") and hasattr(adapter.decoder, "out_fc"):
        for _, mod in linear_modules:
            if mod is adapter.decoder.out_fc:
                return mod
    return linear_modules[-1][1]  # fallback: last linear layer


def _compute_kfac_hessian(adapter, diffusion, loader, curvature, device, target_module):
    """Accumulate KFAC Kronecker factors and return the full (m, m) Hessian matrix."""
    d_in = target_module.in_features
    d_out = target_module.out_features
    in_feat_sz = d_in + (1 if target_module.bias is not None else 0)
    m = in_feat_sz * d_out

    A_sum = torch.zeros((in_feat_sz, in_feat_sz), device=device)
    B_sum = torch.zeros((d_out, d_out), device=device)
    a_cache, g_cache = [], []

    def fw_hook(mod, x_in, y_out):  # noqa: ARG001 — y_out required by hook API
        a = x_in[0].detach()
        if mod.bias is not None:
            a = torch.cat([a, torch.ones(a.shape[0], 1, device=a.device)], dim=1)
        a_cache.append(a)

    def bw_hook(mod, g_in, g_out):  # noqa: ARG001 — mod, g_in required by hook API
        g_cache.append(g_out[0].detach())

    h_fw = target_module.register_forward_hook(fw_hook)
    h_bw = target_module.register_full_backward_hook(bw_hook)
    n_total = 0

    try:
        for x_0 in loader:
            x_0 = x_0.to(device)
            t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=device)
            noise = torch.randn_like(x_0)
            x_t = diffusion.q_sample(x_0, t, noise=noise)
            a_cache.clear()
            g_cache.clear()
            adapter.zero_grad()
            y_pred = adapter(x_t, t)

            if curvature == "ggn":
                a = a_cache[0].clone()
                # Gradient of each output dimension separately to form the GGN.
                # Multiplying by sqrt(2) matches MSE reduction="sum" 2nd derivative
                # being 2 on the diagonal: g * sqrt(2) -> g^2 * 2.
                for k in range(y_pred.shape[1]):
                    adapter.zero_grad()
                    g_cache.clear()
                    y_pred[:, k].sum().backward(retain_graph=True)
                    B_sum += torch.einsum("bi,bj->ij", g_cache[-1], g_cache[-1])
                A_sum += torch.einsum("bi,bj->ij", a, a)
            elif curvature == "ef":
                y = _get_diffusion_target(diffusion=diffusion, x_0=x_0, x_t=x_t, noise=noise, t=t)
                loss = 0.5 * torch.nn.functional.mse_loss(y_pred, y, reduction="sum")
                loss.backward()
                A_sum += torch.einsum("bi,bj->ij", a_cache[0], a_cache[0])
                B_sum += torch.einsum("bi,bj->ij", g_cache[-1], g_cache[-1])
            else:
                raise ValueError(f"Unsupported curvature '{curvature}'. Expected 'ggn' or 'ef'.")

            n_total += x_0.shape[0]
    finally:
        h_fw.remove()
        h_bw.remove()

    H_kfac = 0.5 * torch.kron(B_sum, A_sum) / max(n_total, 1)

    # Permute from KFAC ordering (row-major over (out, in+bias)) to PyTorch
    # parameter ordering (weight rows first, then bias vector).
    idx = torch.zeros(m, dtype=torch.long, device=device)
    for o in range(d_out):
        for c in range(d_in):
            idx[o * in_feat_sz + c] = o * d_in + c
        if target_module.bias is not None:
            idx[o * in_feat_sz + d_in] = d_out * d_in + o

    H = torch.zeros((m, m), device=device)
    H[idx.unsqueeze(1), idx.unsqueeze(0)] = H_kfac
    return H


def _compute_outer_product_hessian(adapter, diffusion, loader, flat_indices, curvature, d, params_dict, buffers_dict, device, full):
    """Accumulate diagonal or full Hessian via squared per-sample Jacobians/gradients."""
    m = len(flat_indices)
    H = torch.zeros((m, m) if full else m, device=device)
    einsum_str = "bp,bq->pq" if full else "bp,bp->p"

    for x_0 in loader:
        x_0 = x_0.to(device)
        t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=device)
        noise = torch.randn_like(x_0)
        x_t = diffusion.q_sample(x_0, t, noise=noise)

        if curvature == "ggn":
            Gs = _ggn_diag_batch(adapter, x_t, t, flat_indices, d, params_dict, buffers_dict)
        elif curvature == "ef":
            y = _get_diffusion_target(diffusion=diffusion, x_0=x_0, x_t=x_t, noise=noise, t=t)
            Gs = _ef_diag_batch(adapter, x_t, y, t, flat_indices, params_dict, buffers_dict)
        else:
            raise ValueError(f"Unsupported curvature '{curvature}'. Expected 'ggn' or 'ef'.")

        H += 0.5 * torch.einsum(einsum_str, Gs.detach(), Gs.detach())

    return H


def _hessian_to_sigma(H, approximation, prior_precision, m, device):
    # Convert accumulated Hessian H to posterior covariance sigma.
    if approximation in {"full", "kfac"}:
        H = (H + H.T) / 2  # ensure exact symmetry before inversion
        return torch.linalg.inv(H + prior_precision * torch.eye(m, device=device))
    elif approximation == "diagonal":
        return 1.0 / (H + prior_precision)
    else:
        raise ValueError(f"Unsupported approximation '{approximation}'. Expected 'full', 'kfac', or 'diagonal'.")


def compute_hessian_approx(adapter, diffusion, real_data, curvature, flat_indices, n_batches, batch_size, prior_precision, device, approximation="diagonal"):
    if curvature not in {"ef", "ggn"}:
        raise ValueError(f"Unsupported curvature '{curvature}'. Use 'ef' or 'ggn'.")
    if approximation not in {"diagonal", "full", "kfac"}:
        raise ValueError(f"Unsupported approximation '{approximation}'. Use 'diagonal', 'full', or 'kfac'.")

    flat_indices = flat_indices.to(device=device, dtype=torch.long)
    m = len(flat_indices)

    # if approximation == "icla":
    #     # Identity Curvature Laplace Approximation: skip data, use flat prior
    #     return torch.ones(m, device=device) / prior_precision

    for p in adapter.parameters():
        p.requires_grad_(True)

    dataset = LaplaceCalibrationDataset(real_data=real_data, total_samples=n_batches * batch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if approximation == "kfac":
        target_module = _find_kfac_target_module(adapter)
        in_feat_sz = target_module.in_features + (1 if target_module.bias is not None else 0)
        expected_m = in_feat_sz * target_module.out_features
        if expected_m != m:
            raise ValueError(f"KFAC size mismatch: module has {expected_m} params but m={m}.")
        H = _compute_kfac_hessian(adapter, diffusion, loader, curvature, device, target_module)
    elif approximation in {"diagonal", "full"}:
        d = real_data.shape[-1]
        params_dict = dict(adapter.named_parameters())
        buffers_dict = dict(adapter.named_buffers())
        H = _compute_outer_product_hessian(
            adapter, diffusion, loader, flat_indices, curvature,
            d, params_dict, buffers_dict, device, full=(approximation == "full"),
        )
    else:
        raise ValueError(f"Unsupported approximation '{approximation}'. Expected 'kfac', 'diagonal', or 'full'.")

    return _hessian_to_sigma(H, approximation, prior_precision, m, device), H
