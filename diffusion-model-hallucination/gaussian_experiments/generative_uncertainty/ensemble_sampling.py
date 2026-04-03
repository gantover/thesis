import os
import numpy as np
import torch
from pathlib import Path
from .ensemble_weights import get_diffusion
from .model_loading import (
    load_deep_ensemble_models,
    load_llla_sampled_models,
    load_lora_ensemble_models,
    load_base_model,
)

def gen_deep_ensemble_samples(num_samples, batch_size, device, samples_cache_dir, trained_models_dir, sel_generation, M):
    models = load_deep_ensemble_models(trained_models_dir=trained_models_dir, sel_generation=sel_generation, M=M, device=device)
    Path(samples_cache_dir).mkdir(parents=True, exist_ok=True)
    samples_cache_path = Path(samples_cache_dir) / "deep_ensemble_samples.npy"
    sample_ensemble_samples(ensemble_models=models, num_samples=num_samples, batch_size=batch_size, device=device, samples_cache_path=samples_cache_path)

def gen_llla_ensemble_samples(num_samples, batch_size, device, samples_cache_dir, llla_sampled_models_dir, trained_models_dir, sel_generation, M):
    base_model = load_base_model(trained_models_dir=trained_models_dir, sel_generation=sel_generation, device=device)
    llla_models = load_llla_sampled_models(llla_sampled_models_dir=llla_sampled_models_dir, M=M, device=device)
    models = [base_model] + llla_models
    Path(samples_cache_dir).mkdir(parents=True, exist_ok=True)
    samples_cache_path = Path(samples_cache_dir) / "llla_ensemble_samples.npy"
    sample_ensemble_samples(ensemble_models=models, num_samples=num_samples, batch_size=batch_size, device=device, samples_cache_path=samples_cache_path)

def gen_lora_ensemble_samples(num_samples, batch_size, device, samples_cache_dir, lora_sampled_models_dir, trained_models_dir, sel_generation, M, r, alpha):
    base_model = load_base_model(trained_models_dir=trained_models_dir, sel_generation=sel_generation, device=device)
    lora_models = load_lora_ensemble_models(lora_sampled_models_dir=lora_sampled_models_dir, base_model=base_model, M=M, r=r, alpha=alpha, device=device)
    models = [base_model] + lora_models
    Path(samples_cache_dir).mkdir(parents=True, exist_ok=True)
    samples_cache_path = Path(samples_cache_dir) / "lora_ensemble_samples.npy"
    sample_ensemble_samples(ensemble_models=models, num_samples=num_samples, batch_size=batch_size, device=device, samples_cache_path=samples_cache_path)

def sample_ensemble_samples(ensemble_models, num_samples, batch_size, device, samples_cache_path):
    timesteps = 1000
    num_models = len(ensemble_models)

    # if os.path.exists(cache_path):
    #     print(f"Loading cached ensemble samples from {cache_path}")
    #     ensemble_samples = np.load(cache_path)

    # create a fixed dataset of pure initial noise
    torch.manual_seed(42)
    pure_noise_dataset = torch.randn((num_samples, 2), device=device)
    
    # diffusion parameters matching the training script
    diffusion = get_diffusion(timesteps=timesteps)

    # pre allocation
    ensemble_samples = np.zeros((num_models, num_samples, 2), dtype=np.float32)
    
    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
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