import copy
import os

import numpy as np
import torch

from ddpm_torch.toy import Decoder


def load_model_from_checkpoint(f_chkpt_dir, device, seed=0, sel_generation=0):
    model = Decoder(in_features=2, mid_features=128, num_temporal_layers=3)
    chkpt_dir = f_chkpt_dir(seed)
    chkpt_path = os.path.join(chkpt_dir, f"ddpm_gaussian25_gen_{sel_generation}.pt")
    if not os.path.exists(chkpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {chkpt_path}")

    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.to(device)
    model.eval()
    return model


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


def build_deep_ensemble_models(
    f_chkpt_dir,
    sel_generation,
    num_additional_models,
    device,
):
    total_models = num_additional_models + 1
    print("Loading ensemble models from independent checkpoints...")
    models = []
    for model_seed in range(total_models):
        models.append(
            load_model_from_checkpoint(
                f_chkpt_dir=f_chkpt_dir,
                device=device,
                seed=model_seed,
                sel_generation=sel_generation,
            )
        )
    return models


def build_last_layer_laplace_models(
    f_chkpt_dir,
    sel_generation,
    num_additional_models,
    diffusion,
    device,
    laplace_batches=64,
    laplace_batch_size=2048,
    prior_precision=1e-2,
    fisher_scale_mode="dataset_size",
    fisher_scale=1.0,
    sample_temperature=1.0,
    weight_sampling_seed=None,
):
    print("Building ensemble by last-layer Laplace weight sampling...")
    base_model = load_model_from_checkpoint(
        f_chkpt_dir=f_chkpt_dir,
        device=device,
        seed=0,
        sel_generation=sel_generation,
    )

    real_dataset_path = f_chkpt_dir(0) + "/real_dataset.npy"
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

    for _ in range(num_additional_models):
        models.append(
            sample_last_layer_model(
                base_model=base_model,
                means=means,
                stds=stds,
                device=device,
                temperature=sample_temperature,
                generator=weight_generator,
            )
        )
    return models
