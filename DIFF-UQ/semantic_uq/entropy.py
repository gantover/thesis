import numpy as np


def exact_gaussian_entropy(mu_array: np.ndarray, sigma_squared: float = 1e-3, anchor_base: bool = True) -> np.ndarray:
    """
    Computes the EXACT full-covariance entropy of a multivariate Gaussian 
    using the Dual Covariance (Gram Matrix) trick to avoid dimensional explosion.
    """
    # np.linalg.eigvalsh does not support float16; promote to float64 for stability.
    mu_array = np.asarray(mu_array, dtype=np.float64)

    if len(mu_array.shape) == 2:
        mu_array = mu_array[np.newaxis, ...]

    B, M, D = mu_array.shape
    entropy = np.empty(B, dtype=np.float64)

    # The number of dimensions where true variance actually exists is at most min(M-1, D)
    active_dims = min(M - 1, D)
    
    # The constant term for the full D-dimensional entropy
    constant_term = 0.5 * D * (np.log(2 * np.pi) + 1)

    for i in range(B):
        samples = mu_array[i]  # Shape: (M, D)
        
        # 1. Base the variation on the clean reference image (index 0).
        # This avoids the variance being dominated by the perturbed images' 
        # drift/bias when M is large, anchoring uncertainty to the true image.
        if anchor_base:
            reference = samples[0:1] # shape (1, D)
        else:
            reference = np.mean(samples, axis=0, keepdims=True) # shape (1, D)
            
        centered = samples - reference  # Shape: (M, D)
        
        # 2. Compute the Dual Covariance (Gram) Matrix. 
        # Shape: (M, M). For M=6, this is a tiny 6x6 matrix!
        # This is mathematically equivalent to computing the DxD pseudo-covariance matrix.
        dual_cov = (centered @ centered.T) / M
        
        # 3. Get the eigenvalues. 
        eigenvalues = np.linalg.eigvalsh(dual_cov)
        
        # 4. A centered M x M matrix has exactly M-1 non-zero eigenvalues.
        # Extract the top active eigenvalues and clip to prevent floating point noise.
        active_eigenvalues = np.sort(eigenvalues)[-active_dims:]
        active_eigenvalues = np.clip(active_eigenvalues, 0.0, None)
        
        # 5. Add observation noise to the active dimensions
        log_det_active = np.sum(np.log(active_eigenvalues + sigma_squared))
        
        # 6. For the remaining (D - active_dims) empty dimensions, 
        # the eigenvalue is exactly 0. So adding noise just makes them sigma_squared.
        log_det_inactive = (D - active_dims) * np.log(sigma_squared)
        
        # 7. Total Log Determinant is the sum of active and inactive dimensions
        total_log_det = log_det_active + log_det_inactive
        
        entropy[i] = 0.5 * total_log_det + constant_term

    if len(mu_array.shape) == 2:
        return entropy[0]
    return entropy


def gaussian_entropy(mu_array: np.ndarray, sigma_squared: float = 1e-3, anchor_base: bool = True) -> np.ndarray:
    """
    Calculate the entropy of multivariate Gaussian distributions with covariance
    Diag(1/M * Σ(μₘ²) - μ̄²) + σ²I in batch mode.
    """
    # Keep computations in at least float32 to avoid precision/compatibility issues.
    mu_array = np.asarray(mu_array, dtype=np.float32)

    if len(mu_array.shape) == 2:
        mu_array = mu_array[np.newaxis, ...]

    _, _, D = mu_array.shape

    # Focus uncertainty on divergence measured from the unperturbed reference image (index 0).
    if anchor_base:
        reference_features = mu_array[:, 0:1, :]
    else:
        reference_features = np.mean(mu_array, axis=1, keepdims=True)
    
    diagonal_terms = np.mean((mu_array - reference_features) ** 2, axis=1)
    diagonal_terms = np.clip(diagonal_terms, 0.0, None)  # because with only M=6 samples there are some negative values
    eigenvalues = diagonal_terms + sigma_squared  # Shape: (N, D)
    log_det = np.sum(np.log(eigenvalues), axis=1)  # Shape: (N,)

    entropy = 0.5 * log_det + 0.5 * D * (np.log(2 * np.pi) + 1)

    if len(mu_array.shape) == 2:
        return entropy[0]
    return entropy

def trace_variance(mu_array: np.ndarray, anchor_base: bool = True) -> np.ndarray:
    """
    Calculate the Total Variance (Trace of the covariance matrix)
    for each item in the batch. This sums the marginal variances 
    and is highly robust to rank-deficiency and anisotropic feature spaces.
    """
    mu_array = np.asarray(mu_array, dtype=np.float32)

    if len(mu_array.shape) == 2:
        mu_array = mu_array[np.newaxis, ...]

    # Variance per dimension relative to the clean reference image (index 0).
    # This prevents the metric from ignoring systematic shifts when M is large.
    if anchor_base:
        reference_features = mu_array[:, 0:1, :]
    else:
        reference_features = np.mean(mu_array, axis=1, keepdims=True)

    diagonal_terms = np.mean((mu_array - reference_features) ** 2, axis=1)
    diagonal_terms = np.clip(diagonal_terms, 0.0, None)
    
    # Trace is simply the sum of all dimensional variances
    trace = np.sum(diagonal_terms, axis=1)

    if len(mu_array.shape) == 2:
        return trace[0]
    return trace
