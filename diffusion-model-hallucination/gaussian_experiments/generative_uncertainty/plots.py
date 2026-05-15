import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_uncertainty_threshold_analysis(uncertainties, percentiles=None, figure_filename=None):
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
    if figure_filename is not None:
        plt.savefig(figure_filename, dpi=300, bbox_inches="tight")
        print(f"saved: {figure_filename}")
    plt.show()

def show_scatter_with_threshold(real_data, uncertainty_scores, base_samples, threshold, show_removed=False):
    percentile_score = jnp.percentile(uncertainty_scores, threshold)
    print(percentile_score)
    confident_mask = uncertainty_scores <= percentile_score
    unconfident_mask = uncertainty_scores > percentile_score
    filtered_samples = base_samples[confident_mask]
    unconfident_samples = base_samples[unconfident_mask]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(real_data[:, 0], real_data[:, 1], s=2, alpha=0.5, color='tab:orange')
    axes[0].set_title("Train Dataset")

    axes[1].scatter(base_samples[:, 0], base_samples[:, 1], s=2, alpha=0.5, color='tab:blue')
    axes[1].set_title("Generated Dataset")

    axes[2].scatter(filtered_samples[:, 0], filtered_samples[:, 1], s=2, alpha=0.5, color='tab:blue')
    if show_removed:
        axes[2].scatter(unconfident_samples[:, 0], unconfident_samples[:, 1], s=2, alpha=0.5, color='tab:red')
    axes[2].set_title("Filtered Dataset")


    for ax in axes:
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        
    plt.tight_layout()
    plt.show()

def show_hallucination_analysis(uncertainty_scores, base_samples, threshold):
    percentile_score = jnp.percentile(uncertainty_scores, threshold)
    print(percentile_score)
    unconfident_mask = uncertainty_scores > percentile_score
    from generative_uncertainty.scoring import extract_true_gmm_params

    # 1. Get ground truth distribution properties
    true_means, true_var = extract_true_gmm_params()
    std_dev = jnp.sqrt(true_var)

    # 2. Define a "Ground Truth Hallucination"
    # e.g., anything further than 4 standard deviations from its closest mode's center
    dist_threshold = 6.0 * std_dev

    # 3. Calculate distance to the closest mode for all generated samples
    diffs = base_samples[:, None, :] - true_means[None, :, :]
    distances = jnp.linalg.norm(diffs, axis=-1)
    min_distances = jnp.min(distances, axis=1)

    # Ground truth labels
    is_true_hallucination = min_distances > dist_threshold

    # Predicted labels from your uncertainty percentile (unconfident_mask)
    is_pred_hallucination = unconfident_mask 

    # 4. Compute Confusion Matrix Masks
    TP_mask = is_true_hallucination & is_pred_hallucination   # Rightfully detected (True Positives)
    FP_mask = (~is_true_hallucination) & is_pred_hallucination  # Wrongfully detected (False Positives - The "Halos")
    TN_mask = (~is_true_hallucination) & (~is_pred_hallucination) # Rightfully kept (True Negatives)
    FN_mask = is_true_hallucination & (~is_pred_hallucination)   # Missed hallucinations (False Negatives)

    # 5. Plot it!
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(base_samples[TN_mask, 0], base_samples[TN_mask, 1], s=15, alpha=0.5, color='tab:blue', 
            label=f'Rightfully Kept (TN): {TN_mask.sum()}')
    ax.scatter(base_samples[FP_mask, 0], base_samples[FP_mask, 1], s=15, alpha=0.5, color='tab:orange', 
            label=f'Wrongly Shaved Halos (FP): {FP_mask.sum()}')
    ax.scatter(base_samples[TP_mask, 0], base_samples[TP_mask, 1], s=20, alpha=0.9, color='tab:green', 
            label=f'Rightfully Caught (TP): {TP_mask.sum()}')
    ax.scatter(base_samples[FN_mask, 0], base_samples[FN_mask, 1], s=20, alpha=0.9, color='tab:red', 
            label=f'Missed Hallucinations (FN): {FN_mask.sum()}')

    # Draw circles around the modes to visualize the threshold
    for mu in true_means:
        circle = plt.Circle((mu[0], mu[1]), dist_threshold, color='black', fill=False, linestyle='--', alpha=0.9, linewidth=1.5)
        ax.add_patch(circle)

    ax.set_title(f"Predicted vs True Hallucinations (at {threshold}%)")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize='large')
    ax.legend(ncol=1, loc='lower center', frameon=True, fancybox=True, shadow=True)
    # ax.legend()
    plt.tight_layout()
    plt.show()

    return {
        "TP": TP_mask,
        "FP": FP_mask,
        "TN": TN_mask,
        "FN": FN_mask
    }
