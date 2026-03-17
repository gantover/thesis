from .jazbec_compute import cache_path, f_uncertainty_scores, f_chkpt_dir, plot_samples_filtering
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ot
from sklearn.metrics.pairwise import rbf_kernel
import scipy.stats as st
from tqdm.auto import tqdm

def main():
    real_dataset_path = f_chkpt_dir(0) + "/real_dataset.npy"
    real_data = np.load(real_dataset_path)
    print(f"loaded real data : {real_data.shape}")

    ensemble_samples = np.load(cache_path)
    print(f"loaded ensemble samples: {ensemble_samples.shape}")
    base_samples = ensemble_samples[0]

    uncertainty_calc_method = "diagonal_gaussian_entropy" # "diagonal_gaussian_entropy", "full_gaussian_entropy", "raw_variance"
    uncertainty_scores = f_uncertainty_scores(ensemble_samples, kind=uncertainty_calc_method)

    df = pd.DataFrame()
    percentiles = [70, 75, 80, 85, 90, 95, 100]
    for percentile in tqdm(percentiles):
        percentile_score = np.percentile(uncertainty_scores, percentile)
        confident_mask = uncertainty_scores <= percentile_score
        filtered_samples = base_samples[confident_mask]
        
        results_filtered = estimator(calculate_wasserstein, real_data, filtered_samples, num_iterations=20)
        df[percentile] = results_filtered
    df.to_pickle("./results/wasserstein_results.pkl")

def mixture_rbf_mmd(X, Y, gammas=[2.0, 10.0, 400.0]):
    """
    Calculates MMD using a mixture of RBF kernels to capture both 
    local and global distribution structures.
    """
    mmd_sum = 0
    for gamma in gammas:
        XX = rbf_kernel(X, X, gamma)
        YY = rbf_kernel(Y, Y, gamma)
        XY = rbf_kernel(X, Y, gamma)
        mmd_sum += XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd_sum

def calculate_wasserstein(real_data, generated_data):
    """Calculates the 2-Wasserstein distance between two 2D distributions."""
    # Subsample to speed up calculation if datasets are huge (e.g., 5000 points)
    assert len(real_data) <= 5000 and len(generated_data) <= 5000, "Subsample the data to at most 5000 points for efficiency."
    assert len(real_data) == len(generated_data)
    n = len(real_data)
    
    # Calculate pairwise Euclidean distance matrix
    M = ot.dist(real_data, generated_data, metric='euclidean')
    
    # Uniform weights for all points
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    
    # Calculate exact Earth Mover's Distance
    wasserstein_dist = ot.emd2(a, b, M)
    # wasserstein_dist = ot.sinkhorn2(a, b, M, reg=0.01, numItermax=10000)[0]
    return wasserstein_dist

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
        
    scores = np.array(scores)
    
    # 3. Calculate Statistical Metrics
    mean_mmd = np.mean(scores)
    std_dev = np.std(scores, ddof=1)  # ddof=1 for unbiased sample standard deviation
    standard_error = std_dev / np.sqrt(num_iterations)
    
    # 4. Calculate 95% Confidence Interval using Student's t-distribution
    ci_lower, ci_upper = st.t.interval(confidence=0.95, df=num_iterations-1, loc=mean_mmd, scale=standard_error)
    
    return {
        "mean": mean_mmd,
        "std_dev": std_dev,
        "standard_error": standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

if __name__ == "__main__":
    main()