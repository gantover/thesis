import numpy as np
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