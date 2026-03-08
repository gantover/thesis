import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from ddpm_torch.toy import Decoder, GaussianDiffusion, get_beta_schedule

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 50000
    batch_size = 10000
    timesteps = 1000
    model_selected = 0
    
    # 1. Create a fixed dataset of pure initial noise
    torch.manual_seed(42)
    # This guarantees every model starts denoising from the exact same coordinates
    pure_noise_dataset = torch.randn((num_samples, 2), device=device)
    
    # Setup diffusion parameters matching the training script
    betas = get_beta_schedule("linear", beta_start=0.001, beta_end=0.2, timesteps=timesteps)
    diffusion = GaussianDiffusion(
        betas=betas, 
        model_mean_type="eps", 
        model_var_type="fixed-large", 
        loss_type="mse"
    )
    
    ensemble_samples = []
    
    # 2. Denoise datapoints with the 6 models
    # We assume models were trained with seeds 1, 2, 3, 4, 5, 6
    for model_seed in range(0,6):
        print(f"Generating samples for model {model_seed}/5...")
        
        model = Decoder(in_features=2, mid_features=128, num_temporal_layers=3)
        
        # Reconstruct the checkpoint directory name 
        chkpt_dir = f"./chkpts/gaussian25_100000_g_5_e_10000_t1000_m128_nl3_blinear_seed{model_seed}_ensemble_model_seed_{model_seed}"
        chkpt_path = os.path.join(chkpt_dir, f"ddpm_gaussian25_gen_{model_selected}.pt")
        
        if not os.path.exists(chkpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {chkpt_path}. Did you train seed {model_seed}?")
            
        checkpoint = torch.load(chkpt_path, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        model.to(device)
        model.eval()
        
        model_samples = []
        
        with torch.no_grad():
            for i, batch_start in enumerate(range(0, num_samples, batch_size)):
                batch_noise = pure_noise_dataset[batch_start:batch_start+batch_size]
                
                # Use the same seed for every model on this batch so that the
                # intermediate denoising noise is identical across ensemble members.
                # This ensures the variance only reflects model disagreement, not
                # sampling stochasticity.
                torch.manual_seed(12345 + i)
                samples = diffusion.p_sample(
                    model, 
                    noise=batch_noise, 
                    device=device, 
                    seed=None
                )
                model_samples.append(samples.cpu().numpy())
                
        ensemble_samples.append(np.concatenate(model_samples, axis=0))
        
    ensemble_samples = np.stack(ensemble_samples, axis=0) # Shape: (6, 50000, 2)
    
    # 3. Evaluate generative uncertainty using the last 5 models
    base_samples = ensemble_samples[0]
    uncertainty_ensemble = ensemble_samples[1:] # Models 1 through 5
    
    # Jazbec et al. calculate the uncertainty as the entropy of the moment-matched 
    # Gaussian approximation. For a diagonal covariance matrix, the entropy is 
    # proportional to the sum of the logs of the variances.
    variances = np.var(uncertainty_ensemble, axis=0) # Shape: (50000, 2)
    uncertainty_scores = 0.5 * np.sum(np.log(variances + 1e-9), axis=1) # Shape: (50000,)
    
    # 4. Filter the 50% lowest generative uncertainty
    median_uncertainty = np.median(uncertainty_scores)
    confident_mask = uncertainty_scores <= median_uncertainty
    filtered_samples = base_samples[confident_mask]
    
    print(f"Original samples generated: {len(base_samples)}")
    print(f"Filtered samples retained:  {len(filtered_samples)}")
    
    # 5. Plot the 3 figures to reproduce Figure 2
    # Attempt to load the real dataset saved by the training script
    real_dataset_path = "./chkpts/gaussian25_100000_g_5_e_10000_t1000_m128_nl3_blinear_seed0_ensemble_model_seed_0/real_dataset.npy"
    if os.path.exists(real_dataset_path):
        real_data = np.load(real_dataset_path)
    else:
        print("Warning: real_dataset.npy not found! Plotting base_samples for the first panel instead.")
        real_data = base_samples # Fallback

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Panel 1: Train Dataset
    axes[0].scatter(real_data[:, 0], real_data[:, 1], s=2, alpha=0.5, color='tab:orange')
    axes[0].set_title("Train Dataset")
    
    # Panel 2: Generated Dataset
    axes[1].scatter(base_samples[:, 0], base_samples[:, 1], s=2, alpha=0.5, color='tab:blue')
    axes[1].set_title("Generated Dataset")
    
    # Panel 3: Filtered Dataset
    axes[2].scatter(filtered_samples[:, 0], filtered_samples[:, 1], s=2, alpha=0.5, color='tab:blue')
    axes[2].set_title("Filtered Dataset")
    
    # Format axes identical to the paper's representation
    for ax in axes:
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        
    plt.tight_layout()
    plt.savefig("generative_uncertainty_figure2.jpg", dpi=300)
    print("Success! Saved reproduction plot to generative_uncertainty_figure2.jpg")

if __name__ == "__main__":
    main()