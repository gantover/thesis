import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from ddpm_torch.toy import GaussianDiffusion, get_beta_schedule
from ensemble_weights import get_diffusion, load_ensemble_models

try:
    from .ensemble_weights import build_deep_ensemble_models, build_last_layer_laplace_models
except ImportError:
    from ensemble_weights import build_deep_ensemble_models, build_last_layer_laplace_models

num_samples = 100000 # samples to generate from each model in the ensemble (including base model)
sel_generation = 0 # which recursive generation to load from the checkpoints
percentile_threshold = 70 # percentile threshold for filtering samples based on uncertainty scores (lower is more strict)
uncertainty_calc_method = "diagonal_gaussian_entropy" # "diagonal_gaussian_entropy", "full_gaussian_entropy", "raw_variance"
# Source of ensemble weights:
# - "deep_ensemble": use checkpoints from seed 0..M
# - "last_layer_laplace": use seed-0 checkpoint + M sampled last-layer weight sets
ensemble_weight_source = "last_layer_laplace" # "deep_ensemble" or "last_layer_laplace"
num_additional_weight_sets = 5 # M additional models on top of the base model
f_chkpt_dir = lambda seed: f"./chkpts/gaussian25_100000_g_1_e_10000_t1000_m128_nl3_blinear_seed{seed}_fixed_ds_ensemble_model_seed_{seed}"
cache_base_dir = "/dtu/blackhole/13/213811/s243425/gaussian_experiment/samples"

    # uncertainty_scores = f_uncertainty_scores(uncertainty_ensemble, kind=uncertainty_calc_method) # Shape: (50000,)
    
    # percentile_score = np.percentile(uncertainty_scores, percentile_threshold)
    # confident_mask = uncertainty_scores <= percentile_score
    # filtered_samples = base_samples[confident_mask]
    
    # print(f"Original samples generated: {len(base_samples)}")
    # print(f"Filtered samples retained:  {len(filtered_samples)}")
    
    # plot_samples_filtering(base_samples, filtered_samples)
    # plot_uncertainty_threshold_analysis(uncertainty_scores)

def f_uncertainty_scores(
    uncertainty_ensemble,
    kind: str = "diagonal_gaussian_entropy",
    eps: float = 1e-8,
):

    if kind == "diagonal_gaussian_entropy":
        # reproduction of Jazbec et al. method from appendix B.1
        # gaussian obtained with moment matching, followed by diagnonal covariance assumption
        variances = np.var(uncertainty_ensemble, axis=0)
        uncertainty_scores = 0.5 * np.sum(np.log(variances + eps), axis=1)

    elif kind == "full_gaussian_entropy":
        # removing the diagonal covariance assumption
        uncertainty_scores = []
        for i in range(len(uncertainty_ensemble[0])): # Iterate over each sample
            points = uncertainty_ensemble[:, i, :] # Shape: (num_models, 2)

            # the full 2x2 covariance matrix
            cov_matrix = np.cov(points, rowvar=False) 
            
            # Calculate the log determinant (adding a small epsilon to the diagonal for stability)
            cov_matrix += np.eye(2) * eps
            sign, logdet = np.linalg.slogdet(cov_matrix)
            
            # Entropy is proportional to log determinant
            uncertainty_scores.append(0.5 * logdet)
            
        uncertainty_scores = np.array(uncertainty_scores)

    elif kind == "raw_variance":
        uncertainty_scores = np.sum(np.var(uncertainty_ensemble, axis=0), axis=1)

    else:
        raise ValueError(f"Unknown uncertainty calculation method: {kind}")

    return uncertainty_scores

def plot_samples_filtering(base_samples, filtered_samples, save_fig=True):
    real_dataset_path = f_chkpt_dir(0) + "/real_dataset.npy"

    if os.path.exists(real_dataset_path):
        real_data = np.load(real_dataset_path)
    else:
        raise FileNotFoundError("real_dataset.npy not found")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(real_data[:, 0], real_data[:, 1], s=2, alpha=0.5, color='tab:orange')
    axes[0].set_title("Train Dataset")
    
    axes[1].scatter(base_samples[:, 0], base_samples[:, 1], s=2, alpha=0.5, color='tab:blue')
    axes[1].set_title("Generated Dataset")

    axes[2].scatter(filtered_samples[:, 0], filtered_samples[:, 1], s=2, alpha=0.5, color='tab:blue')
    axes[2].set_title("Filtered Dataset")

    # for i in range(1, num_models):
    #     axes[i][1].scatter(ensemble_samples[i, :, 0], ensemble_samples[i, :, 1], s=2, alpha=0.5, color='tab:blue')
    
    for ax in axes:
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        
    plt.tight_layout()
    if save_fig:
        figure_filename = f"./figures/samples_{uncertainty_calc_method}_{percentile_threshold}_{num_samples}_{ensemble_weight_source}.jpg"
        plt.savefig(figure_filename, dpi=300)
        print(f"saved reproduction plot to {figure_filename}")
    else:
        plt.show()


def plot_uncertainty_threshold_analysis(uncertainties, percentiles=None):
    # plot the uncertainty distribution with CDF and annotated percentile thresholds.
    if percentiles is None:
        percentiles = [25, 50, 70, 75, 77.5, 80, 90, 95]

    uncertainties = np.array(uncertainties)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(percentiles)))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.hist(uncertainties, bins=100, density=True, alpha=0.4, color="steelblue", label="Density")
    ax1.set_xlabel("Uncertainty Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Uncertainty Distribution with Percentile Thresholds")

    ax1_twin = ax1.twinx()
    sorted_u = np.sort(uncertainties)
    pct_axis = np.linspace(0, 100, len(sorted_u))
    ax1_twin.plot(sorted_u, pct_axis, color="darkorange", linewidth=2, label="Percentile")
    ax1_twin.set_ylabel("Percentile", color="darkorange")
    ax1_twin.tick_params(axis="y", labelcolor="darkorange")
    ax1_twin.set_ylim(0, 105)

    y_top = ax1.get_ylim()[1]
    for p, c in zip(percentiles, colors):
        threshold = np.percentile(uncertainties, p)
        ax1.axvline(threshold, color=c, linestyle="--", linewidth=1.5, label=f"p{p}")
        ax1.text(threshold, y_top * 0.98, f"p{p}", color=c, fontsize=8,
                 ha="center", va="top", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c, alpha=0.8))

    # dummy handles for legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="steelblue", linewidth=8, alpha=0.4, label="Density"),
        Line2D([0], [0], color="darkorange", linewidth=2, label="Percentile"),
    ]
    ax1.legend(handles=handles, fontsize=9, loc="upper left")

    plt.tight_layout()
    figure_filename = f"./figures/uncertainty_distribution_{uncertainty_calc_method}_{num_samples}_{ensemble_weight_source}.png"
    plt.savefig(figure_filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"saved: {figure_filename}")