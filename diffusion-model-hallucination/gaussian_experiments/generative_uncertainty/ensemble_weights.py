import copy
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from laplace.baselaplace import DiagLaplace
from laplace.curvature.curvlinops import CurvlinopsEF

from ddpm_torch.toy import Decoder, GaussianDiffusion, get_beta_schedule


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


class ToyCurvlinopsEF(CurvlinopsEF):
    """DIFF-UQ-style custom EF backend with explicit (x, y, t) signature."""

    def gradients(self, x, y, t):
        def loss_single(x, y, t, params_dict, buffers_dict):
            x, y, t = x.unsqueeze(0), y.unsqueeze(0), t.unsqueeze(0)
            output = torch.func.functional_call(
                self.model,
                (params_dict, buffers_dict),
                (x, t),
            )
            loss = torch.func.functional_call(self.lossfunc, {}, (output, y))
            return loss, loss

        grad_fn = torch.func.grad(loss_single, argnums=3, has_aux=True)
        batch_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, 0, None, None))

        batch_grad, batch_loss = batch_grad_fn(x, y, t, self.params_dict, self.buffers_dict)
        Gs = torch.cat([bg.flatten(start_dim=1) for bg in batch_grad.values()], dim=1)

        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]

        loss = batch_loss.sum(0)
        return Gs, loss

    def diag(self, x, y, t, **kwargs):
        Gs, loss = self.gradients(x, y, t)
        Gs, loss = Gs.detach(), loss.detach()
        diag_ef = torch.einsum("bp,bp->p", Gs, Gs)
        return self.factor * loss, self.factor * diag_ef


class ToyCurvlinopsGGN(CurvlinopsEF):
    """GGN curvature backend: differentiates model output (not loss), no residuals."""

    def gradients(self, x, y, t):
        d = x.shape[-1]
        Gs_list = []
        for k in range(d):
            def _output_k(x, t, params_dict, buffers_dict, _k=k):
                x, t = x.unsqueeze(0), t.unsqueeze(0)
                return torch.func.functional_call(self.model, (params_dict, buffers_dict), (x, t))[0, _k]

            grad_fn = torch.func.grad(_output_k, argnums=2)
            batch_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, None, None))
            batch_grad_k = batch_grad_fn(x, t, self.params_dict, self.buffers_dict)
            flat_k = torch.cat([g.flatten(start_dim=1) for g in batch_grad_k.values()], dim=1)
            Gs_list.append(flat_k)

        Gs = torch.cat(Gs_list, dim=0)  # (B*d, p_total)
        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]
        return Gs, torch.zeros(1, device=x.device)

    def diag(self, x, y, t, **kwargs):
        Gs, _ = self.gradients(x, y, t)
        return 0.0, self.factor * torch.einsum("bp,bp->p", Gs, Gs).detach()


class ToyLLDiagLaplace(DiagLaplace):
    """DIFF-UQ-style DiagLaplace wrapper for toy diffusion calibration data."""

    def __init__(
        self,
        model,
        diffusion,
        last_layer_name,
        backend=ToyCurvlinopsEF,
        likelihood="regression",
        sigma_noise=1.0,
        prior_precision=1.0,
        prior_mean=0.0,
        temperature=1.0,
    ):
        self.diffusion = diffusion

        total_params = 0
        selected_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            keep = last_layer_name in name
            param.requires_grad = keep
            if keep:
                selected_params += param.numel()

        if selected_params == 0:
            raise ValueError(f"No parameters matched last_layer_name='{last_layer_name}'.")

        print(f"Total parameters: {total_params}")
        print(f"Total parameters in selected layer(s): {selected_params}")

        super().__init__(
            model,
            likelihood,
            sigma_noise,
            prior_precision,
            prior_mean,
            temperature,
            backend=backend,
        )

    def _preprocess_batch(self, x_0):
        x_0 = x_0.to(self._device)
        t = torch.randint(0, self.diffusion.timesteps, (x_0.shape[0],), device=self._device)
        noise = torch.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise=noise)

        if self.diffusion.model_mean_type == "mean":
            y = self.diffusion.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)[0]
        elif self.diffusion.model_mean_type == "x_0":
            y = x_0
        elif self.diffusion.model_mean_type == "eps":
            y = noise
        else:
            raise NotImplementedError(self.diffusion.model_mean_type)

        return x_t, y, t

    def fit(self, train_loader, override=True):
        if override:
            self._init_H()
            self.loss = 0
            self.n_data = 0

        self.model.eval()
        self.mean = parameters_to_vector(self.params)
        if not self.enable_backprop:
            self.mean = self.mean.detach()

        x_0 = next(iter(train_loader))
        if isinstance(x_0, (tuple, list)):
            x_0 = x_0[0]

        x_t, y, t = self._preprocess_batch(x_0)
        with torch.no_grad():
            out = self.model(x_t, t)
        out = out.view(out.size(0), -1)
        self.n_outputs = out.shape[-1]
        setattr(self.model, "output_size", self.n_outputs)

        N = len(train_loader.dataset)
        for x_0 in train_loader:
            if isinstance(x_0, (tuple, list)):
                x_0 = x_0[0]

            x_t, y, t = self._preprocess_batch(x_0)
            self.model.zero_grad()
            loss_batch, H_batch = self._curv_closure(x_t, y, t, N)
            self.loss += loss_batch
            self.H += H_batch

        self.n_data += N

    def _curv_closure(self, x_t, y, t, N):
        return self.backend.diag(x_t, y, t, N=N, **self._asdl_fisher_kwargs)


def get_diffusion(timesteps=1000):
    betas = get_beta_schedule("linear", beta_start=0.001, beta_end=0.2, timesteps=timesteps)
    return GaussianDiffusion(
        betas=betas,
        model_mean_type="eps",
        model_var_type="fixed-large",
        loss_type="mse",
    )


def fit_last_layer_diag_laplace(
    model,
    diffusion,
    real_data,
    device,
    laplace_batches=64,
    laplace_batch_size=2048,
    prior_precision=1e-2,
    sample_temperature=1.0,
    last_layer_name="out_fc",
    backend=ToyCurvlinopsEF,
):
    dataset = LaplaceCalibrationDataset(
        real_data=real_data,
        total_samples=laplace_batches * laplace_batch_size,
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=laplace_batch_size, shuffle=False)

    la = ToyLLDiagLaplace(
        model=LaplaceDecoderAdapter(model).to(device),
        diffusion=diffusion,
        last_layer_name=last_layer_name,
        prior_precision=prior_precision,
        temperature=sample_temperature,
        backend=backend,
    )
    la.fit(train_loader)
    return la


def _ef_diag_batch(adapter, x_t, y, t, flat_indices, params_dict, buffers_dict):
    """Per-sample EF squared gradients at selected flat_indices. Returns (B, m) tensor."""
    def loss_single(x, y, t, params_dict, buffers_dict):
        x, y, t = x.unsqueeze(0), y.unsqueeze(0), t.unsqueeze(0)
        output = torch.func.functional_call(adapter, (params_dict, buffers_dict), (x, t))
        # Match laplace-torch regression scaling (MSELoss with reduction='sum').
        loss = torch.nn.functional.mse_loss(output, y, reduction="sum")
        return loss

    grad_fn = torch.func.grad(loss_single, argnums=3)
    batch_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, 0, None, None))
    batch_grad = batch_grad_fn(x_t, y, t, params_dict, buffers_dict)
    Gs = torch.cat([g.flatten(start_dim=1) for g in batch_grad.values()], dim=1)  # (B, p_total)
    return Gs[:, flat_indices]  # (B, m)


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

    Gs = torch.cat(Gs_list, dim=0)  # (B*d, p_total)
    return Gs[:, flat_indices]  # (B*d, m)


def _get_diffusion_target(diffusion, x_0, x_t, noise, t):
    """Training target for the current diffusion mean parameterization."""
    if diffusion.model_mean_type == "eps":
        return noise
    if diffusion.model_mean_type == "x_0":
        return x_0
    return diffusion.q_posterior_mean_var(x_0=x_0, x_t=x_t, t=t)[0]


def _cap_posterior_sigma_by_std(sigma, max_posterior_std):
    """Clip posterior std to avoid numerically unstable sampled networks in very flat directions."""
    if max_posterior_std is None:
        return sigma, 0.0

    if max_posterior_std <= 0:
        raise ValueError(f"max_posterior_std must be > 0 or None, got {max_posterior_std}.")

    std = sigma.sqrt()
    clipped = std > max_posterior_std
    clipped_frac = clipped.float().mean().item()
    if clipped.any():
        std = std.clamp(max=max_posterior_std)
    return std.square(), clipped_frac


def _effective_max_std_for_subnetwork(max_posterior_std, m_eff, std_reference_subnetwork_size):
    """Scale max posterior std as sqrt(m_ref / m_eff) for large random subnetworks.

    This keeps the expected perturbation norm from exploding as m increases.
    For m_eff <= m_ref, no extra scaling is applied.
    """
    if max_posterior_std is None:
        return None

    if std_reference_subnetwork_size is None:
        return max_posterior_std

    if std_reference_subnetwork_size <= 0:
        raise ValueError(
            f"std_reference_subnetwork_size must be > 0 or None, got {std_reference_subnetwork_size}."
        )

    scale = min(1.0, math.sqrt(float(std_reference_subnetwork_size) / float(m_eff)))
    return max_posterior_std * scale


def compute_diag_hessian(adapter, diffusion, real_data, curvature, flat_indices, n_batches, batch_size, prior_precision, device):
    """Compute diagonal Hessian for a random subnetwork and return sigma=(H+prior)^{-1}.

    This mirrors laplace-torch conventions used in the last-layer path:
    - regression scaling factor 0.5,
    - data-summed curvature (no extra 1/N averaging here).
    """
    if curvature not in {"ef", "ggn"}:
        raise ValueError(f"Unsupported curvature '{curvature}'. Use 'ef' or 'ggn'.")

    flat_indices = flat_indices.to(device=device, dtype=torch.long)

    # Ensure all params have requires_grad for functional_call / vmap to differentiate them
    for p in adapter.parameters():
        p.requires_grad_(True)

    dataset = LaplaceCalibrationDataset(real_data=real_data, total_samples=n_batches * batch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    params_dict = dict(adapter.named_parameters())
    buffers_dict = dict(adapter.named_buffers())
    d = real_data.shape[-1]
    m = len(flat_indices)

    H_diag = torch.zeros(m, device=device)
    # For regression likelihood, laplace-torch curvature backends use factor=0.5.
    hessian_factor = 0.5

    for x_0 in loader:
        x_0 = x_0.to(device)
        t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=device)
        noise = torch.randn_like(x_0)
        x_t = diffusion.q_sample(x_0, t, noise=noise)

        if curvature == "ggn":
            Gs = _ggn_diag_batch(adapter, x_t, t, flat_indices, d, params_dict, buffers_dict)
        else:
            y = _get_diffusion_target(diffusion=diffusion, x_0=x_0, x_t=x_t, noise=noise, t=t)
            Gs = _ef_diag_batch(adapter, x_t, y, t, flat_indices, params_dict, buffers_dict)

        # Sum curvature contributions over data points; no additional 1/N averaging.
        H_diag += hessian_factor * torch.einsum("bp,bp->p", Gs.detach(), Gs.detach())

    sigma = 1.0 / (H_diag + prior_precision)
    return sigma


def sample_last_layer_model(base_model, sampled_layer_vector, device, last_layer_name="out_fc"):
    sampled_model = copy.deepcopy(base_model)
    selected_params = [p for name, p in sampled_model.named_parameters() if last_layer_name in name]
    if not selected_params:
        raise ValueError(f"No parameters matched last_layer_name='{last_layer_name}'.")

    with torch.no_grad():
        vector_to_parameters(sampled_layer_vector.to(selected_params[0].device), selected_params)

    sampled_model.to(device)
    sampled_model.eval()
    return sampled_model


def sample_subset_model(base_model, sampled_vector, flat_indices, device):
    """Inject sampled_vector at flat_indices positions into a deep copy of base_model."""
    sampled = copy.deepcopy(base_model)
    flat = parameters_to_vector(sampled.parameters()).detach().clone()
    flat[flat_indices.to(flat.device)] = sampled_vector.to(flat.device)
    vector_to_parameters(flat, sampled.parameters())
    sampled.to(device)
    sampled.eval()
    return sampled


def load_model_from_checkpoint(chkpt_path, device):
    model = Decoder(in_features=2, mid_features=128, num_temporal_layers=3)

    # Local experiment checkpoints are trusted and may require full unpickling.
    try:
        checkpoint = torch.load(chkpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(chkpt_path, map_location=device)

    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.to(device)
    model.eval()
    return model


def load_deep_ensemble_models(
    trained_models_dir,
    sel_generation,
    M,
    device,
):
    total_models = M + 1
    print("Loading ensemble models from independent checkpoints...")
    models = []
    for model_seed in range(total_models):
        chkpt_dir = Path(trained_models_dir.format(seed=model_seed))
        chkpt_path = chkpt_dir / f"ddpm_gaussian25_gen_{sel_generation}.pt"
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {chkpt_path}")
        models.append(load_model_from_checkpoint(chkpt_path=chkpt_path, device=device))
    return models


def load_llla_sampled_models(llla_sampled_models_dir, M, device):
    models = []
    for model_id in range(M):
        chkpt_path = Path(llla_sampled_models_dir) / f"llla_sample_{model_id}.pt"
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Missing LLLA sampled model checkpoint: {chkpt_path}")
        models.append(load_model_from_checkpoint(chkpt_path=chkpt_path, device=device))
    return models


def load_base_model(trained_models_dir, sel_generation, device):
    chkpt_dir = Path(trained_models_dir.format(seed=0))
    chkpt_path = chkpt_dir / f"ddpm_gaussian25_gen_{sel_generation}.pt"
    if not chkpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {chkpt_path}")
    return load_model_from_checkpoint(chkpt_path=chkpt_path, device=device)


def build_last_layer_laplace_models(
    trained_models_dir,
    llla_sampled_models_dir,
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
):
    print("Building ensemble by last-layer Laplace weight sampling...")

    chkpt_dir = Path(trained_models_dir.format(seed=0))
    base_model = load_base_model(
        trained_models_dir=trained_models_dir,
        sel_generation=sel_generation,
        device=device,
    )

    real_dataset_path = chkpt_dir / "real_dataset.npy"
    if not os.path.exists(real_dataset_path):
        raise FileNotFoundError(f"Missing calibration data: {real_dataset_path}")
    real_data = np.load(real_dataset_path)

    la = fit_last_layer_diag_laplace(
        model=base_model,
        diffusion=diffusion,
        real_data=real_data,
        device=device,
        laplace_batches=laplace_batches,
        laplace_batch_size=laplace_batch_size,
        prior_precision=prior_precision,
        sample_temperature=sample_temperature,
        last_layer_name=last_layer_name,
    )

    if weight_sampling_seed is not None:
        torch.manual_seed(weight_sampling_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(weight_sampling_seed)

    sampled_layers = la.sample(M)
    if sampled_layers.ndim == 1:
        sampled_layers = sampled_layers.unsqueeze(0)

    llla_chkpt_dir = Path(llla_sampled_models_dir)
    llla_chkpt_dir.mkdir(parents=True, exist_ok=True)

    models = [base_model]
    for i in range(M):
        sampled_model = sample_last_layer_model(
            base_model=base_model,
            sampled_layer_vector=sampled_layers[i],
            device=device,
            last_layer_name=last_layer_name,
        )
        model_cache_path = llla_chkpt_dir / f"llla_sample_{i}.pt"
        torch.save(sampled_model.state_dict(), model_cache_path)
        print(f"Saved sampled model to cache: {model_cache_path}")
        models.append(sampled_model)

    return models


def build_laplace_ensemble(
    trained_models_dir,
    llla_sampled_models_dir,
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
    subset="last_layer",    # 'last_layer' | 'random'
    curvature="ef",         # 'ef' | 'ggn'
    m=1000,                 # subnetwork size (only for subset='random')
    subset_seed=42,
    max_posterior_std=1.0,
    std_reference_subnetwork_size=1000,
):
    """Factory for MC ensemble via Laplace posterior.

    subset='last_layer': fits DiagLaplace on last affine layer only (uses laplace-torch internals).
    subset='random':     fits diagonal Hessian on m random parameters across all layers.
    curvature='ef':      empirical Fisher (gradient of loss).
    curvature='ggn':     generalized Gauss-Newton (gradient of output, no residuals).
    """
    print(f"Building Laplace ensemble: subset={subset}, curvature={curvature}")

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

    llla_chkpt_dir = Path(llla_sampled_models_dir)
    llla_chkpt_dir.mkdir(parents=True, exist_ok=True)

    if subset == "last_layer":
        backend_cls = ToyCurvlinopsGGN if curvature == "ggn" else ToyCurvlinopsEF
        la = fit_last_layer_diag_laplace(
            model=base_model,
            diffusion=diffusion,
            real_data=real_data,
            device=device,
            laplace_batches=laplace_batches,
            laplace_batch_size=laplace_batch_size,
            prior_precision=prior_precision,
            sample_temperature=sample_temperature,
            last_layer_name=last_layer_name,
            backend=backend_cls,
        )
        sampled_layers = la.sample(M)
        if sampled_layers.ndim == 1:
            sampled_layers = sampled_layers.unsqueeze(0)

        models = [base_model]
        for i in range(M):
            sampled_model = sample_last_layer_model(
                base_model=base_model,
                sampled_layer_vector=sampled_layers[i],
                device=device,
                last_layer_name=last_layer_name,
            )
            model_cache_path = llla_chkpt_dir / f"llla_sample_{i}.pt"
            torch.save(sampled_model.state_dict(), model_cache_path)
            print(f"Saved sampled model: {model_cache_path}")
            models.append(sampled_model)

    else:  # subset == 'random'
        # Use a deep copy for the adapter so that vmap/functional_call TensorWrapper state
        # does not leak back into base_model (which must stay deepcopy-able for sampling).
        adapter = LaplaceDecoderAdapter(copy.deepcopy(base_model)).to(device)
        total_params = sum(p.numel() for p in adapter.parameters())
        if m <= 0:
            raise ValueError(f"m must be > 0, got {m}.")
        m_eff = min(m, total_params)
        if m_eff < m:
            print(f"Warning: requested m={m} exceeds total_params={total_params}; using m={m_eff}.")

        rng = torch.Generator()
        rng.manual_seed(subset_seed)
        flat_indices = torch.randperm(total_params, generator=rng)[:m_eff].to(device)

        sigma = compute_diag_hessian(
            adapter=adapter,
            diffusion=diffusion,
            real_data=real_data,
            curvature=curvature,
            flat_indices=flat_indices,
            n_batches=laplace_batches,
            batch_size=laplace_batch_size,
            prior_precision=prior_precision,
            device=device,
        )
        # sigma shape: (m,). Scale by temperature^2 (legacy behavior used in this project).
        sigma = torch.clamp(sigma, min=1e-12) * (sample_temperature ** 2)

        eff_max_std = _effective_max_std_for_subnetwork(
            max_posterior_std=max_posterior_std,
            m_eff=m_eff,
            std_reference_subnetwork_size=std_reference_subnetwork_size,
        )
        sigma, clipped_frac = _cap_posterior_sigma_by_std(
            sigma=sigma,
            max_posterior_std=eff_max_std,
        )
        print(
            "Random subset posterior sigma stats: "
            f"min={sigma.min().item():.3e}, median={sigma.median().item():.3e}, max={sigma.max().item():.3e}, "
            f"std_clip_frac={clipped_frac:.3f}, effective_max_std={eff_max_std}"
        )

        # Sample M weight vectors from N(mean, diag(sigma)).
        # Use base_model (not adapter) to get the mean — same parameter ordering, adapter is now tainted.
        mean_subset = parameters_to_vector(base_model.parameters()).detach()[flat_indices]
        eps = torch.randn(M, m_eff, device=device)
        sampled_layers = mean_subset.unsqueeze(0) + eps * sigma.sqrt().unsqueeze(0)

        models = [base_model]
        for i in range(M):
            sampled_model = sample_subset_model(
                base_model=base_model,
                sampled_vector=sampled_layers[i],
                flat_indices=flat_indices,
                device=device,
            )
            model_cache_path = llla_chkpt_dir / f"llla_sample_{i}.pt"
            torch.save(sampled_model.state_dict(), model_cache_path)
            print(f"Saved sampled model: {model_cache_path}")
            models.append(sampled_model)

    return models
