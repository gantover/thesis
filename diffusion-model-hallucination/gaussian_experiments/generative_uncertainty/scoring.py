import numpy as np
import jax.numpy as jnp
import pandas as pd
# import ot
from sklearn.metrics.pairwise import rbf_kernel
import scipy.stats as st
from tqdm.auto import tqdm

from scipy.special import logsumexp
from scipy.stats import entropy

def gmm_score_percentiles(real_data, ensemble_samples, uncertainty_scores, percentiles=None, percentile_step=1):
    true_means, true_var = extract_true_gmm_params()

    df = pd.DataFrame()
    if percentiles is None:
        percentiles = list(np.arange(70, 101, percentile_step))

    for percentile in tqdm(percentiles):
        percentile_score = jnp.percentile(uncertainty_scores, percentile)
        confident_mask = uncertainty_scores <= percentile_score
        filtered_samples = ensemble_samples[0][confident_mask]
        
        results_filtered = evaluate_exact_gmm(filtered_samples, true_means, true_var)
        df[percentile] = results_filtered

    return df

def mmd_score_percentiles(real_data, ensemble_samples, uncertainty_scores, percentiles=None, gamma=0.25, num_iterations=20, subsample_size=5000):
    df = pd.DataFrame()

    if percentiles is None:
        percentiles = [70, 75, 80, 85, 90, 95, 100]

    for percentile in tqdm(percentiles):
        percentile_score = jnp.percentile(uncertainty_scores, percentile)
        confident_mask = uncertainty_scores <= percentile_score
        filtered_samples = ensemble_samples[0][confident_mask]
        
        results_filtered = estimator(lambda X, Y: single_rbf_mmd(X, Y, gamma=gamma), real_data, filtered_samples, num_iterations=num_iterations, subsample_size=subsample_size)
        df[percentile] = results_filtered

    return df

def single_rbf_mmd(X, Y, gamma=None):
    """
    MMD with a single RBF kernel.
    If gamma is None, it is set by the median heuristic: 1 / (2 * median(||xi - xj||^2)).
    For Gaussian25, the analytical value is ~0.25.
    """
    if gamma is None:
        from sklearn.metrics.pairwise import euclidean_distances
        dists_sq = euclidean_distances(X, squared=True)
        median_sq = np.median(dists_sq[np.triu_indices(len(X), k=1)])
        gamma = 1.0 / (2.0 * median_sq)
        print(f"Using median heuristic for gamma: {gamma:.4f}")
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def estimator(metric, real_data, generated_data, num_iterations=20, subsample_size=5000):
    """
    Performs Monte Carlo estimation of MMD with variance and error bounds.
    """
    print(f"Running {num_iterations} Monte Carlo iterations")
    scores = []
    
    # Safely handle the case where the filtered dataset is smaller than subsample_size
    gen_sample_size = min(subsample_size, len(generated_data))
    
    for i in tqdm(range(num_iterations), desc="Monte Carlo Iterations"):
        # 1. Draw independent random subsamples
        idx_real = np.random.choice(len(real_data), subsample_size, replace=False)
        idx_gen = np.random.choice(len(generated_data), gen_sample_size, replace=False)
        
        real_sub = real_data[idx_real]
        gen_sub = generated_data[idx_gen]
        
        # 2. Calculate the chosen metric for this iteration
        score = metric(real_sub, gen_sub)
        scores.append(score)
        
    scores = jnp.array(scores)
    
    # 3. Calculate Statistical Metrics
    mean_mmd = jnp.mean(scores)
    std_dev = jnp.std(scores, ddof=1)  # ddof=1 for unbiased sample standard deviation
    standard_error = std_dev / jnp.sqrt(num_iterations)
    
    # 4. Calculate 95% Confidence Interval using Student's t-distribution
    ci_lower, ci_upper = st.t.interval(confidence=0.95, df=num_iterations-1, loc=mean_mmd, scale=standard_error)
    
    return {
        "mean": mean_mmd,
        "std_dev": std_dev,
        "standard_error": standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

def extract_true_gmm_params(scale=2.0, noise=0.05):
    """
    Extracts the EXACT mathematically true scaled means and variance
    based on the Gaussian25 class in toy_data.py.
    """
    # 1. The unscaled modes
    base_modes = np.array([(i, j) for i in range(-2, 3) for j in range(-2, 3)], dtype=np.float32)
    modes = scale * base_modes
    
    # 2. The exact scaling factor used in the dataset
    stdev_scale = np.sqrt(noise ** 2 + (scale ** 2) * 2.)
    
    # 3. The true scaled parameters
    true_means = modes / stdev_scale
    true_variance = (noise / stdev_scale) ** 2
    
    return true_means, true_variance

def evaluate_within_mode_variance(generated_data, true_means, true_variance):
    """
    Measures whether generated samples reproduce the correct within-mode spread.
    Assigns each generated point to its nearest mode, computes empirical variance
    around that mode center, and compares to the known true variance.

    Returns variance_ratio: 1.0 = perfect, <1 = under-dispersed, >1 = over-dispersed.
    """
    diff = generated_data[:, np.newaxis, :] - true_means[np.newaxis, :, :]  # (N, 25, 2)
    dist_sq = np.sum(diff ** 2, axis=2)                                      # (N, 25)
    assignments = np.argmin(dist_sq, axis=1)                                 # (N,)

    per_mode_var = []
    for k in range(len(true_means)):
        pts = generated_data[assignments == k]
        if len(pts) > 1:
            per_mode_var.append(np.mean(np.var(pts - true_means[k], axis=0)))

    empirical_var = np.mean(per_mode_var)
    return {
        "empirical_variance": empirical_var,
        "variance_ratio": empirical_var / true_variance,
    }

def evaluate_exact_gmm(generated_data, true_means, true_variance):
    """
    Evaluates generated data against the known analytical 25-GMM.
    Returns Exact Log-Likelihood (Precision) and Mode KL Divergence (Recall).
    """
    num_modes = len(true_means)
    N = len(generated_data)
    
    # --- 1. EXACT LOG-LIKELIHOOD (PRECISION) ---
    # We use the log-sum-exp trick for numerical stability
    # log p(x) = log(1/25) + logsumexp( log N(x | mu_i, var) )
    
    # Calculate squared Euclidean distances to all 25 means for all points
    # Shape: (N, 25)
    diff = generated_data[:, np.newaxis, :] - true_means[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Log of the Gaussian PDF: -0.5 * (d^2 / var + 2*log(2*pi*var))
    log_gaussian_pdfs = -0.5 * (dist_sq / true_variance + 2 * np.log(2 * np.pi * true_variance))
    
    # Combine the 25 modes: log( sum(exp(log_pdfs)) ) - log(25)
    log_likelihoods = logsumexp(log_gaussian_pdfs, axis=1) - np.log(num_modes)
    
    # The final Precision score is the average log-likelihood
    avg_log_likelihood = np.mean(log_likelihoods)
    
    # --- 2. EXACT MODE COVERAGE (RECALL) ---
    # Assign each point to the nearest true mean
    closest_mode_idx = np.argmin(dist_sq, axis=1)
    
    # Count how many points went to each mode
    mode_counts = np.bincount(closest_mode_idx, minlength=num_modes)
    empirical_probs = mode_counts / N
    
    # The ideal distribution is perfectly uniform (1/25 for all modes)
    ideal_probs = np.ones(num_modes) / num_modes
    
    # Calculate KL Divergence between Empirical and Ideal
    # Lower is better (0.0 means perfect mode coverage)
    kl_divergence = entropy(empirical_probs, ideal_probs)
    
    # Calculate raw coverage (How many modes have at least 0.5% of the points?)
    covered_modes = np.sum(empirical_probs > 0.005)

    within_mode_variance_results = evaluate_within_mode_variance(generated_data, true_means, true_variance)
    
    return {
        "avg_log_likelihood": avg_log_likelihood,  # Higher is better
        "mode_kl_divergence": kl_divergence,       # Lower is better (0 = perfect)
        "modes_covered": covered_modes,            # Higher is better (Max 25)
        "mode_distribution": empirical_probs,       # For plotting/debugging
        "empirical_variance": within_mode_variance_results['empirical_variance'],  # For checking spread
        "variance_ratio": within_mode_variance_results['variance_ratio'],            # For checking spread
    }