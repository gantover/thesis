import copy
import os

from pathlib import Path
import numpy as np
import torch

from ddpm_torch.toy import Decoder, GaussianDiffusion, get_beta_schedule

def get_diffusion(timesteps=1000):
    betas = get_beta_schedule("linear", beta_start=0.001, beta_end=0.2, timesteps=timesteps)
    diffusion = GaussianDiffusion(
        betas=betas, 
        model_mean_type="eps", 
        model_var_type="fixed-large", 
        loss_type="mse"
    )
    return diffusion



def fit_last_layer_diag_laplace(
    model,
    diffusion,
    real_data,
    device,
    laplace_batches=64,
    laplace_batch_size=2048,
    prior_precision=1e-2,
    fisher_scale_mode="dataset_size",
    fisher_scale=1.0,
    eps=1e-8,
):
    # Fit a diagonal Laplace posterior only on the final linear layer.
    layer = model.out_fc
    params = [layer.weight]
    if layer.bias is not None:
        params.append(layer.bias)

    means = [p.detach().flatten().clone() for p in params]
    precisions = [torch.full_like(m, prior_precision) for m in means]

    data = torch.from_numpy(real_data).float()
    n = len(data)
    num_grad_samples = 0
    for _ in range(laplace_batches):
        idx = torch.randint(0, n, (laplace_batch_size,))
        x_0 = data[idx].to(device)
        t = torch.randint(0, diffusion.timesteps, (laplace_batch_size,), device=device)
        noise = torch.randn_like(x_0)

        losses = diffusion.train_losses(model, x_0=x_0, t=t, noise=noise)
        for bi in range(losses.shape[0]):
            retain = bi < losses.shape[0] - 1
            grads = torch.autograd.grad(losses[bi], params, retain_graph=retain, create_graph=False)
            for i, g in enumerate(grads):
                precisions[i] += g.detach().flatten() ** 2
            num_grad_samples += 1

    if num_grad_samples == 0:
        raise RuntimeError("No gradients accumulated for Laplace fit.")

    # Match DIFF-UQ style scaling by using mean empirical Fisher then scaling by N.
    # This makes posterior scale less sensitive to the sampling budget used for fit.
    fisher_multiplier = fisher_scale
    if fisher_scale_mode == "dataset_size":
        fisher_multiplier *= n
    elif fisher_scale_mode == "none":
        fisher_multiplier *= 1.0
    else:
        raise ValueError(f"Unknown fisher_scale_mode: {fisher_scale_mode}")

    for i in range(len(precisions)):
        empirical_fisher_mean = (precisions[i] - prior_precision) / num_grad_samples
        precisions[i] = prior_precision + fisher_multiplier * empirical_fisher_mean

    stds = [(p + eps).rsqrt() for p in precisions]
    return means, stds


def sample_last_layer_model(base_model, means, stds, device, temperature=1.0, generator=None):
    sampled_model = copy.deepcopy(base_model)
    layer = sampled_model.out_fc

    with torch.no_grad():
        w_noise = torch.randn(
            stds[0].shape,
            generator=generator,
            device=stds[0].device,
            dtype=stds[0].dtype,
        )
        w_sample = means[0] + temperature * stds[0] * w_noise
        layer.weight.copy_(w_sample.view_as(layer.weight))

        if layer.bias is not None and len(means) > 1:
            b_noise = torch.randn(
                stds[1].shape,
                generator=generator,
                device=stds[1].device,
                dtype=stds[1].dtype,
            )
            b_sample = means[1] + temperature * stds[1] * b_noise
            layer.bias.copy_(b_sample.view_as(layer.bias))

    sampled_model.to(device)
    sampled_model.eval()
    return sampled_model

def load_model_from_checkpoint(chkpt_path, device):
    model = Decoder(in_features=2, mid_features=128, num_temporal_layers=3)

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
        models.append(
            load_model_from_checkpoint(
                chkpt_path=chkpt_path,
                device=device,
            )
        )
    return models

def load_llla_sampled_models(llla_sampled_models_dir, M, device):
    models = []
    total_models = M # we are not loading the base model here, only the sampled ones
    for model_id in range(total_models):
        chkpt_path = Path(llla_sampled_models_dir) / f"llla_sample_{model_id}.pt"
        if not chkpt_path.exists():
            raise FileNotFoundError(f"Missing LLLA sampled model checkpoint: {chkpt_path}")
        models.append(
            load_model_from_checkpoint(
                chkpt_path=chkpt_path,
                device=device,
            )
        )
    return models

def load_base_model(trained_models_dir, sel_generation, device):
    chkpt_dir = Path(trained_models_dir.format(seed=0))
    chkpt_path = chkpt_dir / f"ddpm_gaussian25_gen_{sel_generation}.pt"
    if not chkpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {chkpt_path}")
    return load_model_from_checkpoint(
        chkpt_path=chkpt_path,
        device=device,
    )


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
    fisher_scale_mode="dataset_size",
    fisher_scale=1.0,
    sample_temperature=1.0,
    weight_sampling_seed=None,
):
    print("Building ensemble by last-layer Laplace weight sampling...")

    # loading the base model from the first checkpoint (seed 0) to fit the Laplace posterior
    chkpt_dir = Path(trained_models_dir.format(seed=0))

    base_model = load_base_model(trained_models_dir=trained_models_dir, sel_generation=sel_generation, device=device)

    real_dataset_path = chkpt_dir / "real_dataset.npy"
    if not os.path.exists(real_dataset_path):
        raise FileNotFoundError(f"Missing calibration data: {real_dataset_path}")
    real_data = np.load(real_dataset_path)

    means, stds = fit_last_layer_diag_laplace(
        model=base_model,
        diffusion=diffusion,
        real_data=real_data,
        device=device,
        laplace_batches=laplace_batches,
        laplace_batch_size=laplace_batch_size,
        prior_precision=prior_precision,
        fisher_scale_mode=fisher_scale_mode,
        fisher_scale=fisher_scale,
    )

    models = [base_model]
    weight_generator = None
    if weight_sampling_seed is not None:
        weight_generator = torch.Generator(device=device).manual_seed(weight_sampling_seed)
    
    llla_chkpt_dir = Path(llla_sampled_models_dir)
    llla_chkpt_dir.mkdir(parents=True, exist_ok=True)

    for i in range(M):
        model = sample_last_layer_model(
                    base_model=base_model,
                    means=means,
                    stds=stds,
                    device=device,
                    temperature=sample_temperature,
                    generator=weight_generator,
                )
        model_cache_path = llla_chkpt_dir / f"llla_sample_{i}.pt"
        torch.save(model.state_dict(), model_cache_path)
        print(f"Saved sampled model to cache: {model_cache_path}")
        models.append(model)
    return models
