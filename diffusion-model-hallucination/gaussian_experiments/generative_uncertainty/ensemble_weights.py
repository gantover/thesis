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
        for i, x_0 in enumerate(train_loader):
            print(i)
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
    )
    la.fit(train_loader)
    return la


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
