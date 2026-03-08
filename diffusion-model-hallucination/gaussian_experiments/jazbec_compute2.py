import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from ddpm_torch.toy import Decoder, GaussianDiffusion, get_beta_schedule

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 100000
    batch_size = 10000
    timesteps = 1000
    model_selected = 0
    num_models = 6 # 1 base model + 5 ensemble models
    
    # create a fixed dataset of pure initial noise
    torch.manual_seed(42)
    pure_noise_dataset = torch.randn((num_samples, 2), device=device)
    
    # diffusion parameters matching the training script
    betas = get_beta_schedule("linear", beta_start=0.001, beta_end=0.2, timesteps=timesteps)
    diffusion = GaussianDiffusion(
        betas=betas, 
        model_mean_type="eps", 
        model_var_type="fixed-large", 
        loss_type="mse"
    )
    
    # pre allocation
    ensemble_samples = np.zeros((num_models, num_samples, 2), dtype=np.float32)
    
    print("Loading ensemble models into memory...")
    ensemble_models = []
    for model_seed in range(num_models):
        model = Decoder(in_features=2, mid_features=128, num_temporal_layers=3)

        # Reconstruct the checkpoint directory name 
        num_gens = 5 
        # lbs = "lbs_"
        lbs = ""
        chkpt_dir = f"./chkpts/gaussian25_100000_g_{num_gens}_e_10000_t1000_m128_nl3_blinear_seed{model_seed}_{lbs}ensemble_model_seed_{model_seed}"
        chkpt_path = os.path.join(chkpt_dir, f"ddpm_gaussian25_gen_{model_selected}.pt")
        
        if not os.path.exists(chkpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {chkpt_path}. Did you train seed {model_seed}?")
            
        checkpoint = torch.load(chkpt_path, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        model.to(device)
        model.eval()
        ensemble_models.append(model)
        
    print("Starting generation...")
    
    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_noise = pure_noise_dataset[batch_start:batch_end]
            
            batch_seed = 12345 + batch_start
            print(f"Processing batch {batch_start} to {batch_end}...")
            
            for m_idx, model in enumerate(ensemble_models):
                # CRITICAL STEP: Rewind the RNG for every model in the ensemble.
                # This guarantees that the intermediate denoising noise is IDENTICAL 
                # for all 6 models on this specific batch.
                torch.manual_seed(batch_seed)
                
                samples = diffusion.p_sample(
                    model, 
                    noise=batch_noise, 
                    device=device, 
                    seed=None
                )
                
                ensemble_samples[m_idx, batch_start:batch_end] = samples.cpu().numpy()
                
    base_samples = ensemble_samples[0] # 0 (base model)
    uncertainty_ensemble = ensemble_samples[1:] # 1 -> 5
    
    # variances = np.var(uncertainty_ensemble, axis=0) # Shape: (50000, 2)
    # uncertainty_scores = 0.5 * np.sum(np.log(variances + 1e-8), axis=1) # Shape: (50000,)
    variances = np.var(uncertainty_ensemble, axis=0) # Shape: (50000, 2)
    uncertainty_scores = np.sum(variances, axis=1)   # Shape: (50000,)
    
    # filter 50% lowest generative uncertainty
    median_uncertainty = np.median(uncertainty_scores)
    # percentile_25 = np.percentile(uncertainty_scores, 25)
    confident_mask = uncertainty_scores <= median_uncertainty
    # confident_mask = uncertainty_scores <= percentile_25 
    filtered_samples = base_samples[confident_mask]
    
    print(f"Original samples generated: {len(base_samples)}")
    print(f"Filtered samples retained:  {len(filtered_samples)}")
    
    real_dataset_path = "./chkpts/gaussian25_100000_g_5_e_10000_t1000_m128_nl3_blinear_seed0_ensemble_model_seed_0/real_dataset.npy"
    if os.path.exists(real_dataset_path):
        real_data = np.load(real_dataset_path)
    else:
        raise FileNotFoundError("real_dataset.npy not found")

    fig, axes = plt.subplots(num_models, 3, figsize=(12, 4*num_models))
    
    axes[0][0].scatter(real_data[:, 0], real_data[:, 1], s=2, alpha=0.5, color='tab:orange')
    axes[0][0].set_title("Train Dataset")
    
    axes[0][1].scatter(base_samples[:, 0], base_samples[:, 1], s=2, alpha=0.5, color='tab:blue')
    axes[0][1].set_title("Generated Dataset")

    for i in range(1, num_models):
        axes[i][1].scatter(ensemble_samples[i, :, 0], ensemble_samples[i, :, 1], s=2, alpha=0.5, color='tab:blue')
    
    axes[0][2].scatter(filtered_samples[:, 0], filtered_samples[:, 1], s=2, alpha=0.5, color='tab:blue')
    axes[0][2].set_title("Filtered Dataset")
    
    for i in range(0, num_models):
        for ax in axes[i]:
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 1.8)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        
    plt.tight_layout()
    plt.savefig("generative_uncertainty_figure_new.jpg", dpi=300)
    print("Success! Saved reproduction plot to generative_uncertainty_figure2.jpg")

if __name__ == "__main__":
    main()