import os
import numpy as np
import torch
from pathlib import Path
from .utils import get_param_str_la, get_param_str_la_lora
from .ensemble_weights import get_diffusion
from .model_loading import (
    load_deep_ensemble_models,
    load_la_sampled_models,
    load_lora_ensemble_models,
    load_la_lora_sampled_models,
    load_base_model,
)
from .config import (
    AppConfig, 
    LaplaceEnsembleConfig, 
    LoraEnsembleConfig, 
    OftEnsembleConfig,
    DeepEnsembleConfig, 
    SamplingConfig, 
    LaplaceLoraEnsembleConfig
)

def gen_deep_ensemble_samples(sampling_config: SamplingConfig, device: torch.device, deep_ensemble_config: DeepEnsembleConfig):
    models = load_deep_ensemble_models(de_config=deep_ensemble_config, device=device)
    Path(sampling_config.samples_cache_dir).mkdir(parents=True, exist_ok=True)
    samples_cache_path = Path(sampling_config.samples_cache_dir) / "deep_ensemble_samples.npy"
    sample_ensemble_samples(
        ensemble_models=models,
        sampling_config=sampling_config,
        device=device,
        samples_cache_path=samples_cache_path,
    )

# def gen_la_ensemble_samples(num_samples, batch_size, device, samples_cache_dir, la_sampled_models_dir, trained_models_dir, sel_generation, M, prior_precision, approximation, curvature, subset, m, sample_temperature):
def gen_la_ensemble_samples(sampling_config: SamplingConfig, device: torch.device, la_ensemble_config: LaplaceEnsembleConfig, deep_ensemble_config: DeepEnsembleConfig):
    base_model = load_base_model(de_config=deep_ensemble_config, device=device)
    la_models = load_la_sampled_models(la_config=la_ensemble_config, device=device, de_config=deep_ensemble_config)
    models = [base_model] + la_models
    Path(sampling_config.samples_cache_dir).mkdir(parents=True, exist_ok=True)
    param_str = get_param_str_la(la_ensemble_config)
    samples_cache_path = Path(sampling_config.samples_cache_dir) / f"la_ensemble_samples_{param_str}.npy"
    sample_ensemble_samples(
        ensemble_models=models,
        sampling_config=sampling_config,
        device=device,
        samples_cache_path=samples_cache_path,
    )

def gen_la_lora_ensemble_samples(sampling_config: SamplingConfig, device: torch.device, laplace_lora_config: LaplaceLoraEnsembleConfig, deep_ensemble_config: DeepEnsembleConfig):
    base_model = load_base_model(de_config=deep_ensemble_config, device=device)
    lora_models = load_la_lora_sampled_models(la_lora_config=laplace_lora_config, device=device, de_config=deep_ensemble_config)
    models = [base_model] + lora_models
    Path(sampling_config.samples_cache_dir).mkdir(parents=True, exist_ok=True)
    param_str = get_param_str_la_lora(laplace_lora_config)
    samples_cache_path = Path(sampling_config.samples_cache_dir) / f"la_lora_ensemble_samples_{param_str}.npy"
    sample_ensemble_samples(
        ensemble_models=models,
        sampling_config=sampling_config,
        device=device,
        samples_cache_path=samples_cache_path,
    )

def gen_lora_ensemble_samples(sampling_config: SamplingConfig, device: torch.device, lora_ensemble_config: LoraEnsembleConfig, deep_ensemble_config: DeepEnsembleConfig):
    base_model = load_base_model(de_config=deep_ensemble_config, device=device)
    lora_models = load_lora_ensemble_models(lora_config=lora_ensemble_config, de_config=deep_ensemble_config, device=device)
    models = [base_model] + lora_models
    Path(sampling_config.samples_cache_dir).mkdir(parents=True, exist_ok=True)
    samples_cache_path = Path(sampling_config.samples_cache_dir) / "lora_ensemble_samples.npy"
    sample_ensemble_samples(
        ensemble_models=models,
        sampling_config=sampling_config,
        device=device,
        samples_cache_path=samples_cache_path,
    )


def gen_oft_ensemble_samples(sampling_config: SamplingConfig, device: torch.device, oft_ensemble_config: OftEnsembleConfig, deep_ensemble_config: DeepEnsembleConfig):
    from .model_loading import load_oft_ensemble_models
    base_model = load_base_model(de_config=deep_ensemble_config, device=device)
    oft_models = load_oft_ensemble_models(oft_config=oft_ensemble_config, de_config=deep_ensemble_config, device=device)
    models = [base_model] + oft_models
    Path(sampling_config.samples_cache_dir).mkdir(parents=True, exist_ok=True)
    samples_cache_path = Path(sampling_config.samples_cache_dir) / "oft_ensemble_samples.npy"
    sample_ensemble_samples(
        ensemble_models=models,
        sampling_config=sampling_config,
        device=device,
        samples_cache_path=samples_cache_path,
    )

def sample_ensemble_samples(ensemble_models, sampling_config: SamplingConfig, device, samples_cache_path):
    timesteps = 1000
    num_models = len(ensemble_models)

    # create a fixed dataset of pure initial noise
    torch.manual_seed(42)
    pure_noise_dataset = torch.randn((sampling_config.num_samples, 2), device=device)
    
    # diffusion parameters matching the training script
    diffusion = get_diffusion(timesteps=timesteps)

    # pre allocation
    ensemble_samples = np.zeros((num_models, sampling_config.num_samples, 2), dtype=np.float32)
    
    with torch.no_grad():
        for batch_start in range(0, sampling_config.num_samples, sampling_config.batch_size):
            batch_end = min(batch_start + sampling_config.batch_size, sampling_config.num_samples)
            batch_noise = pure_noise_dataset[batch_start:batch_end]
            
            batch_seed = 12345 + batch_start
            print(f"Processing batch {batch_start} to {batch_end}...")
            
            for m_idx, model in enumerate(ensemble_models):
                # RNG rewinding for every model in the ensemble.
                # guarantees identical intermediate denoising noise.
                torch.manual_seed(batch_seed)
                
                samples = diffusion.p_sample(
                    model, 
                    noise=batch_noise, 
                    device=device, 
                    seed=None
                )
                
                ensemble_samples[m_idx, batch_start:batch_end] = samples.cpu().numpy()

    np.save(samples_cache_path, ensemble_samples)
    print(f"Saved ensemble samples to {samples_cache_path}")