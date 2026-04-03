"""FLARE: Fisher-Laplace Randomized Epistemic uncertainty estimator.

Implements Jacobian-propagated epistemic trace scoring (Gupta et al., arXiv:2602.09170)
for the toy 2D Gaussian diffusion experiment.

Per-sample epistemic trace recursion (Eq. 5 in the paper):
    Sigma^ep_{t-1} = a_t^2 * Sigma^ep_t + b_t^2 * J_{t,I} diag(sigma) J_{t,I}^T

where:
    a_t = 1 / sqrt(1 - betas[t])
    b_t = betas[t] / (sqrt(1 - betas[t]) * sqrt_one_minus_alphas_bar[t])
    sigma = posterior variance of the selected m parameters (shape (m,))
    J_{t,I} = Jacobian rows for output dims, columns for selected params -> shape (d, m)
    trace = sum_k sum_j sigma_j * (d eps_k / d theta_j)^2

The score returned is epi_trace (a scalar per sample), larger = more uncertain.

Raw Decoder is used for Jacobians (TorchScript is compatible with standard autograd).
LaplaceDecoderAdapter is used only during Hessian fitting.
"""

import torch
import numpy as np

from .ensemble_weights import (
    LaplaceDecoderAdapter,
    _cap_posterior_sigma_by_std,
    _effective_max_std_for_subnetwork,
    compute_diag_hessian,
    get_diffusion,
    load_base_model,
)


class FlareLaplace:
    """Diagonal Laplace posterior + FLARE Jacobian-propagated epistemic scoring.

    Args:
        model:            Raw Decoder (NOT wrapped in LaplaceDecoderAdapter).
        diffusion:        GaussianDiffusion instance.
        subset:           'last_layer' or 'random'.
        curvature:        'ef' or 'ggn'.
        m:                Subnetwork size (only for subset='random').
        prior_precision:  Prior precision for diagonal Laplace posterior.
        last_layer_name:  Name prefix of last-layer parameters (for subset='last_layer').
        seed:             RNG seed for random subset selection (should match MC ensemble).
    """

    def __init__(
        self,
        model,
        diffusion,
        subset="last_layer",
        curvature="ef",
        m=1000,
        prior_precision=1e-2,
        last_layer_name="out_fc",
        seed=42,
        max_posterior_std=1.0,
        std_reference_subnetwork_size=1000,
    ):
        self.model = model
        self.diffusion = diffusion
        self.subset = subset
        self.curvature = curvature
        self.prior_precision = prior_precision
        self.last_layer_name = last_layer_name
        self.max_posterior_std = max_posterior_std
        self.std_reference_subnetwork_size = std_reference_subnetwork_size
        self.d = 2  # output dimension for toy experiment

        # Precompute DDPM reverse-step coefficients
        betas = diffusion.betas.float()
        alphas = 1.0 - betas
        self._betas = betas
        self._a = 1.0 / alphas.sqrt()  # a_t = 1/sqrt(alpha_t)
        self._b = betas / (alphas.sqrt() * diffusion.sqrt_one_minus_alphas_bar.float())

        # Select which parameters to score Jacobians against
        if subset == "last_layer":
            self.grad_params = [p for name, p in model.named_parameters() if last_layer_name in name]
            self.flat_indices = None  # means: use the selected params as-is
        else:
            all_params = list(model.parameters())
            total = sum(p.numel() for p in all_params)
            rng = torch.Generator()
            rng.manual_seed(seed)
            self.flat_indices = torch.randperm(total, generator=rng)[:m]
            self.grad_params = all_params  # need grads for all, then index

        self.sigma = None  # set by fit()

    def fit(self, real_data, device, n_batches=64, batch_size=2048):
        """Fit diagonal Laplace posterior; store sigma = (H_diag + prior)^{-1}."""
        adapter = LaplaceDecoderAdapter(self.model).to(device)

        if self.subset == "last_layer":
            # Flat indices for last-layer params relative to adapter's full parameter vector
            flat_idx = []
            offset = 0
            for name, p in adapter.named_parameters():
                if self.last_layer_name in name:
                    flat_idx.append(torch.arange(offset, offset + p.numel()))
                offset += p.numel()
            flat_indices = torch.cat(flat_idx)
        else:
            flat_indices = self.flat_indices

        sigma = compute_diag_hessian(
            adapter=adapter,
            diffusion=self.diffusion,
            real_data=real_data,
            curvature=self.curvature,
            flat_indices=flat_indices,
            n_batches=n_batches,
            batch_size=batch_size,
            prior_precision=self.prior_precision,
            device=device,
        )
        eff_max_std = self.max_posterior_std
        if self.subset == "random":
            eff_max_std = _effective_max_std_for_subnetwork(
                max_posterior_std=self.max_posterior_std,
                m_eff=len(flat_indices),
                std_reference_subnetwork_size=self.std_reference_subnetwork_size,
            )

        sigma, clipped_frac = _cap_posterior_sigma_by_std(
            sigma=sigma,
            max_posterior_std=eff_max_std,
        )
        self.sigma = sigma.to(device)
        self._fit_flat_indices = flat_indices.to(device)
        print(
            "FLARE fit done. sigma: "
            f"min={sigma.min():.4e}, max={sigma.max():.4e}, mean={sigma.mean():.4e}, "
            f"std_clip_frac={clipped_frac:.3f}, effective_max_std={eff_max_std}"
        )

    def score(self, n_samples, device):
        """Generate n_samples from the reverse process and return (samples, epi_traces).

        Returns:
            samples:     np.ndarray of shape (n_samples, 2)
            epi_traces:  np.ndarray of shape (n_samples,)
        """
        if self.sigma is None:
            raise RuntimeError("Call fit() before score().")

        samples = []
        scores = []
        for _ in range(n_samples):
            x0, epi = self._flare_trajectory(device)
            samples.append(x0.cpu().numpy())
            scores.append(epi)

        return np.stack(samples), np.array(scores)

    def _flat_grad(self, device):
        """Collect gradients from grad_params into a flat vector, then index by flat_indices if needed."""
        grads = []
        for p in self.grad_params:
            if p.grad is None:
                grads.append(torch.zeros(p.numel(), device=device))
            else:
                grads.append(p.grad.detach().flatten())
        flat = torch.cat(grads)
        if self.flat_indices is not None:
            flat = flat[self.flat_indices.to(device)]
        return flat

    def _flare_trajectory(self, device):
        """Run one full reverse trajectory; return (x0, epi_trace)."""
        T = self.diffusion.timesteps
        x_t = torch.randn(1, self.d, device=device)
        epi_trace = 0.0

        for ti in range(T - 1, -1, -1):
            t_batch = torch.tensor([ti], device=device)

            # Enable grad on selected params only
            for p in self.grad_params:
                p.requires_grad_(True)
            self.model.zero_grad()

            out = self.model(x_t.detach(), t_batch)  # (1, d)
            eps_t = out.detach()

            # Accumulate squared Jacobian contribution: sum_k (d out_k / d theta_I)^2 * sigma_I
            g_sq = torch.zeros(len(self.sigma), device=device)
            for k in range(self.d):
                if k > 0:
                    self.model.zero_grad()
                out[0, k].backward(retain_graph=(k < self.d - 1))
                g = self._flat_grad(device)
                g_sq = g_sq + g.pow(2)

            for p in self.grad_params:
                p.requires_grad_(False)

            a_t = self._a[ti].item()
            b_t = self._b[ti].item()
            delta = (self.sigma * g_sq.detach()).sum().item()
            epi_trace = a_t ** 2 * epi_trace + b_t ** 2 * delta

            with torch.no_grad():
                mean = a_t * x_t - b_t * eps_t
                if ti > 0:
                    noise_std = self._betas[ti].sqrt().item()
                    x_t = mean + noise_std * torch.randn_like(x_t)
                else:
                    x_t = mean

        return x_t.squeeze(0), epi_trace


def generate_flare_scores(
    n_score_samples,
    device,
    flare_samples_cache_dir,
    trained_models_dir,
    sel_generation,
    real_data_path,
    subset="last_layer",
    curvature="ef",
    m=1000,
    prior_precision=1e-2,
    last_layer_name="out_fc",
    seed=42,
    max_posterior_std=1.0,
    std_reference_subnetwork_size=1000,
    n_batches=64,
    batch_size=2048,
):
    """Top-level function: fit FLARE posterior and generate scored samples.

    Saves two files to flare_samples_cache_dir:
        flare_samples.npy   — shape (n_score_samples, 2)
        flare_scores.npy    — shape (n_score_samples,)
    """
    import os
    from pathlib import Path

    base_model = load_base_model(trained_models_dir=trained_models_dir, sel_generation=sel_generation, device=device)

    real_data = np.load(real_data_path)

    flare = FlareLaplace(
        model=base_model,
        diffusion=get_diffusion(),
        subset=subset,
        curvature=curvature,
        m=m,
        prior_precision=prior_precision,
        last_layer_name=last_layer_name,
        seed=seed,
        max_posterior_std=max_posterior_std,
        std_reference_subnetwork_size=std_reference_subnetwork_size,
    )
    flare.fit(real_data=real_data, device=device, n_batches=n_batches, batch_size=batch_size)

    print(f"Generating {n_score_samples} FLARE-scored samples...")
    samples, scores = flare.score(n_score_samples, device)

    cache_dir = Path(flare_samples_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "flare_samples.npy", samples)
    np.save(cache_dir / "flare_scores.npy", scores)
    print(f"Saved FLARE outputs to {cache_dir}")
    return samples, scores
