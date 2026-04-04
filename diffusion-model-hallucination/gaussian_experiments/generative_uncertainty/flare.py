"""FLARE implementation for toy Gaussian diffusion experiments.

This module implements Algorithm 1 from Gupta et al.:
1) Build a random (or layer-restricted) subnetwork posterior
   Sigma_sub = (H_I + lambda I)^(-1), where H_I is an empirical Fisher block.
2) Run reverse denoising while propagating epistemic uncertainty with
   Sigma_ep(t-1) = a_t^2 Sigma_ep(t) + b_t^2 Delta_t,
   Delta_t = J_t Sigma_sub J_t^T.

For speed, Jacobians are batched with torch.func (vmap + jacrev).
"""

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap
from tqdm.auto import tqdm

from .model_loading import load_base_model
from .ensemble_weights import get_diffusion


class FLAREEstimator:
    """Fisher-Laplace Randomized Estimator for toy denoisers."""

    def __init__(
        self,
        model: torch.nn.Module,
        diffusion,
        subset: str = "random",
        m: int = 512,
        damping: float = 1e-4,
        seed: int = 42,
        last_layer_name: str = "out_fc",
        max_posterior_std: float | None = 1.0,
        delta_clip_max: float = 1e12,
        score_clip_max: float = 1e20,
    ):
        if m <= 0:
            raise ValueError(f"m must be > 0, got {m}.")
        if damping <= 0:
            raise ValueError(f"damping must be > 0, got {damping}.")
        if max_posterior_std is not None and max_posterior_std <= 0:
            raise ValueError("max_posterior_std must be > 0 or None.")
        if delta_clip_max <= 0 or score_clip_max <= 0:
            raise ValueError("delta_clip_max and score_clip_max must be > 0.")

        self.model = model
        self.diffusion = diffusion
        self.subset = subset
        self.m = m
        self.damping = damping
        self.seed = seed
        self.last_layer_name = last_layer_name
        self.max_posterior_std = max_posterior_std
        self.delta_clip_max = delta_clip_max
        self.score_clip_max = score_clip_max

        self.model.eval()

        self._param_names = [name for name, _ in self.model.named_parameters()]
        self._param_numels = [param.numel() for _, param in self.model.named_parameters()]
        self._total_params = int(sum(self._param_numels))

        self._subnetwork_indices = self._sample_subnetwork_indices()
        self.sigma_sub = None

        # FLARE recursion coefficients.
        betas = self.diffusion.betas.float()
        alphas = (1.0 - betas).float()
        alpha_bars = torch.cumprod(alphas, dim=0)
        self._a = torch.rsqrt(alphas)
        self._eps_coef = betas / torch.sqrt(torch.clamp(1.0 - alpha_bars, min=1e-12))
        self._b = betas / (torch.sqrt(alphas) * torch.sqrt(torch.clamp(1.0 - alpha_bars, min=1e-12)))

    def _sample_subnetwork_indices(self) -> torch.Tensor:
        if self.subset == "all":
            return torch.arange(self._total_params, dtype=torch.long)

        if self.subset == "last_layer":
            indices = []
            offset = 0
            for name, param in self.model.named_parameters():
                numel = param.numel()
                if self.last_layer_name in name:
                    indices.extend(range(offset, offset + numel))
                offset += numel
            if not indices:
                raise ValueError(
                    f"No parameters matched last_layer_name='{self.last_layer_name}'."
                )
            return torch.tensor(indices, dtype=torch.long)

        if self.subset == "multi_layer":
            indices = []
            offset = 0
            layer_names = [name.strip() for name in self.last_layer_name.split(",")]
            for name, param in self.model.named_parameters():
                numel = param.numel()
                if any(layer_name in name for layer_name in layer_names):
                    indices.extend(range(offset, offset + numel))
                offset += numel
            if not indices:
                raise ValueError(f"No parameters matched names={layer_names}.")
            return torch.tensor(indices, dtype=torch.long)

        if self.subset == "random":
            m_eff = min(self.m, self._total_params)
            generator = torch.Generator().manual_seed(self.seed)
            return torch.randperm(self._total_params, generator=generator)[:m_eff].sort().values

        raise ValueError(
            f"Unknown subset '{self.subset}'. Use one of ['random', 'last_layer', 'multi_layer', 'all']."
        )

    def _get_functional_state(self, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        params = {
            name: param.detach().to(device).requires_grad_(True)
            for name, param in self.model.named_parameters()
        }
        buffers = {
            name: buf.detach().to(device)
            for name, buf in self.model.named_buffers()
        }
        return params, buffers

    def _flatten_jacobian(self, jacobian_pytree: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for name in self._param_names:
            jac = jacobian_pytree[name]
            parts.append(jac.reshape(jac.shape[0], jac.shape[1], -1))
        return torch.cat(parts, dim=-1)

    def fit(
        self,
        real_data: np.ndarray,
        device: torch.device,
        n_batches: int = 64,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """Fit subnetwork covariance Sigma_sub = (H_I + lambda I)^(-1)."""
        if n_batches <= 0 or batch_size <= 0:
            raise ValueError("n_batches and batch_size must be > 0.")

        data = torch.as_tensor(real_data, dtype=torch.float32, device=device)
        if data.ndim == 1:
            data = data.unsqueeze(1)

        params, buffers = self._get_functional_state(device)
        sub_idx = self._subnetwork_indices.to(device)
        m_eff = int(sub_idx.numel())

        def _forward_single(p, b, x, t):
            return functional_call(self.model, (p, b), (x.unsqueeze(0), t.unsqueeze(0))).squeeze(0)

        jacobian_fn = vmap(jacrev(_forward_single, argnums=0), in_dims=(None, None, 0, 0))

        h_sub = torch.zeros((m_eff, m_eff), dtype=torch.float64, device=device)
        count = 0

        self.model.eval()
        for _ in tqdm(range(n_batches), desc="FLARE Fisher", leave=False):
            batch_ids = torch.randint(0, data.shape[0], (batch_size,), device=device)
            x_0 = data[batch_ids]
            t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=device)
            noise = torch.randn_like(x_0)
            x_t = self.diffusion.q_sample(x_0, t, noise=noise)

            with torch.enable_grad():
                jac_pytree = jacobian_fn(params, buffers, x_t, t)

            jac = self._flatten_jacobian(jac_pytree)[:, :, sub_idx].detach().to(torch.float64)
            jac = torch.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)
            h_sub += torch.einsum("bdm,bdn->mn", jac, jac)
            count += x_t.shape[0]

        h_sub /= float(max(count, 1))

        eye = torch.eye(m_eff, device=device, dtype=torch.float64)
        damped = h_sub + self.damping * eye
        try:
            l_factor = torch.linalg.cholesky(damped)
            sigma_sub = torch.cholesky_solve(eye, l_factor)
        except RuntimeError:
            sigma_sub = torch.linalg.pinv(damped)

        # Optional global scaling to avoid unrealistically large posterior variance.
        if self.max_posterior_std is not None:
            max_var = float(self.max_posterior_std) ** 2
            diag = torch.diag(sigma_sub)
            diag = torch.nan_to_num(diag, nan=max_var, posinf=max_var, neginf=max_var)
            max_diag = torch.max(diag)
            if torch.isfinite(max_diag) and max_diag > max_var:
                sigma_sub = sigma_sub * (max_var / max_diag)

        diag_after = torch.diag(sigma_sub)
        diag_after = torch.nan_to_num(diag_after, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            "FLARE Sigma_sub stats: "
            f"diag_min={diag_after.min().item():.4e}, "
            f"diag_max={diag_after.max().item():.4e}, "
            f"diag_mean={diag_after.mean().item():.4e}"
        )

        self.sigma_sub = sigma_sub.detach()
        return self.sigma_sub

    def sample_and_score(
        self,
        n_samples: int,
        sample_shape: Tuple[int, ...],
        device: torch.device,
        batch_size: int = 128,
        tail_steps: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample x_0 and FLARE trace scores.

        Args:
            n_samples: total number of generated samples.
            sample_shape: tuple like (2,) for 2D toy or (1,) for 1D toy.
            batch_size: score batch size; use modest values to keep Jacobians cheap.
            tail_steps: if > 0, only accumulate FLARE over timesteps t < tail_steps.
        """
        if self.sigma_sub is None:
            raise RuntimeError("Call fit() before sample_and_score().")
        if n_samples <= 0 or batch_size <= 0:
            raise ValueError("n_samples and batch_size must be > 0.")
        if tail_steps < 0:
            raise ValueError("tail_steps must be >= 0.")

        params, buffers = self._get_functional_state(device)
        sub_idx = self._subnetwork_indices.to(device)
        sigma_sub = self.sigma_sub.to(device)

        a = self._a.to(device).to(torch.float64)
        b = self._b.to(device).to(torch.float64)
        eps_coef = self._eps_coef.to(device)

        def _forward_single(p, bfr, x, t):
            return functional_call(self.model, (p, bfr), (x.unsqueeze(0), t.unsqueeze(0))).squeeze(0)

        jacobian_fn = vmap(jacrev(_forward_single, argnums=0), in_dims=(None, None, 0, 0))

        out_samples = []
        out_scores = []

        self.model.eval()
        for start in tqdm(range(0, n_samples, batch_size), desc="FLARE scoring", leave=False):
            current_bs = min(batch_size, n_samples - start)
            x_t = torch.randn((current_bs,) + sample_shape, device=device)
            epistemic_trace = torch.zeros(current_bs, dtype=torch.float64, device=device)

            for ti in range(self.diffusion.timesteps - 1, -1, -1):
                t = torch.full((current_bs,), ti, dtype=torch.long, device=device)

                with torch.no_grad():
                    eps_pred = self.model(x_t, t)

                do_accumulate = (tail_steps == 0) or (ti < tail_steps)
                if do_accumulate:
                    with torch.enable_grad():
                        jac_pytree = jacobian_fn(params, buffers, x_t, t)
                    jac = self._flatten_jacobian(jac_pytree)[:, :, sub_idx].detach().to(torch.float64)
                    jac = torch.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)

                    delta_t = torch.einsum("bdm,mn,bdn->b", jac, sigma_sub, jac)
                    delta_t = torch.nan_to_num(
                        delta_t,
                        nan=0.0,
                        posinf=self.delta_clip_max,
                        neginf=0.0,
                    )
                    delta_t = torch.clamp(delta_t, min=0.0, max=self.delta_clip_max)
                    epistemic_trace = (a[ti] ** 2) * epistemic_trace + (b[ti] ** 2) * delta_t
                    epistemic_trace = torch.nan_to_num(
                        epistemic_trace,
                        nan=0.0,
                        posinf=self.score_clip_max,
                        neginf=0.0,
                    )
                    epistemic_trace = torch.clamp(epistemic_trace, min=0.0, max=self.score_clip_max)

                # Deterministic reverse update (FLARE Algorithm 1, x-hat recursion).
                x_t = a[ti] * (x_t - eps_coef[ti] * eps_pred)

            out_samples.append(x_t.detach().cpu().numpy())
            out_scores.append(epistemic_trace.detach().cpu().numpy())

        return np.concatenate(out_samples, axis=0), np.concatenate(out_scores, axis=0)


def generate_flare_scores(
    n_score_samples,
    device,
    flare_samples_cache_dir,
    trained_models_dir,
    sel_generation,
    real_data_path,
    subset="random",
    m=512,
    prior_precision=1e-4,
    last_layer_name="out_fc",
    seed=42,
    n_batches=64,
    batch_size=512,
    score_batch_size=128,
    tail_steps=0,
    max_posterior_std=1.0,
    delta_clip_max=1e12,
    score_clip_max=1e20,
    **_,
):
    """Fit FLARE and persist scored samples.

    Extra keyword args are ignored so this remains compatible with older config files.
    """
    base_model = load_base_model(
        trained_models_dir=trained_models_dir,
        sel_generation=sel_generation,
        device=device,
    )
    diffusion = get_diffusion()
    real_data = np.load(real_data_path)

    flare = FLAREEstimator(
        model=base_model,
        diffusion=diffusion,
        subset=subset,
        m=m,
        damping=prior_precision,
        seed=seed,
        last_layer_name=last_layer_name,
        max_posterior_std=max_posterior_std,
        delta_clip_max=delta_clip_max,
        score_clip_max=score_clip_max,
    )
    flare.fit(real_data=real_data, device=device, n_batches=n_batches, batch_size=batch_size)

    if real_data.ndim == 1:
        sample_shape = (1,)
    else:
        sample_shape = (int(real_data.shape[1]),)

    samples, scores = flare.sample_and_score(
        n_samples=n_score_samples,
        sample_shape=sample_shape,
        device=device,
        batch_size=score_batch_size,
        tail_steps=tail_steps,
    )

    cache_dir = Path(flare_samples_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "flare_samples.npy", samples)
    np.save(cache_dir / "flare_scores.npy", scores)
    finite_frac = float(np.isfinite(scores).mean())
    print(
        f"Saved FLARE outputs to {cache_dir} | "
        f"score min={np.min(scores):.4e}, max={np.max(scores):.4e}, finite_frac={finite_frac:.3f}"
    )

    return samples, scores
