import numpy as np


def exact_gaussian_entropy(mu_array: np.ndarray, sigma_squared: float = 1e-8, anchor_base: bool = False) -> np.ndarray:
    """
    Computes the EXACT full-covariance entropy of a multivariate Gaussian 
    using the Dual Covariance (Gram Matrix) trick.
    
    Note on sigma_squared: Default is 1e-8 to act as a safe numerical stabilizer 
    for both unnormalized features and L2-normalized embeddings (which have tiny variances).
    """
    mu_array = np.asarray(mu_array, dtype=np.float64)
    if len(mu_array.shape) == 2:
        mu_array = mu_array[np.newaxis, ...]

    B, M, D = mu_array.shape
    entropy = np.empty(B, dtype=np.float64)

    # A centered M-sample matrix has exactly M-1 non-zero eigenvalues.
    active_dims = min(M - 1, D)
    constant_term = 0.5 * D * (np.log(2 * np.pi) + 1)

    for i in range(B):
        samples = mu_array[i]  # Shape: (M, D)
        
        if anchor_base:
            reference = samples[0:1] 
        else:
            reference = np.mean(samples, axis=0, keepdims=True)
            
        centered = samples - reference
        
        # FIX: Use Bessel's correction (M-1) for an unbiased covariance estimator.
        # If anchor_base is True, we are measuring E[(Z-z0)^2], so dividing by M is technically correct for the raw sum of squares, 
        # but for true covariance estimation (anchor_base=False), M-1 is required.
        denominator = M if anchor_base else (M - 1)
        dual_cov = (centered @ centered.T) / denominator
        
        eigenvalues = np.linalg.eigvalsh(dual_cov)
        active_dims = max(0, min(M - 1, D))
        
        if active_dims > 0:
            active_eigenvalues = np.sort(eigenvalues)[-active_dims:]
            active_eigenvalues = np.clip(active_eigenvalues, 0.0, None)
        else:
            active_eigenvalues = np.array([])
        
        log_det_active = np.sum(np.log(active_eigenvalues + sigma_squared))
        log_det_inactive = (D - active_dims) * np.log(sigma_squared)
        total_log_det = log_det_active + log_det_inactive
        
        entropy[i] = 0.5 * total_log_det + constant_term

    if len(mu_array.shape) == 2:
        return entropy[0]
    return entropy

def gaussian_entropy(mu_array: np.ndarray, sigma_squared: float = 1e-8, anchor_base: bool = False) -> np.ndarray:
    """
    Calculate the entropy of diagonal multivariate Gaussian distributions.
    
    Uses sigma_squared=1e-8 as a low-noise floor to accommodate both 
    large unnormalized features and the small variances of L2-normalized features.
    """
    mu_array = np.asarray(mu_array, dtype=np.float32)
    if len(mu_array.shape) == 2:
        mu_array = mu_array[np.newaxis, ...]

    B, M, D = mu_array.shape

    if anchor_base:
        reference_features = mu_array[:, 0:1, :]
        # Mean squared distance to the anchor
        diagonal_terms = np.mean((mu_array - reference_features) ** 2, axis=1)
    else:
        # FIX: Use numpy's built-in variance with ddof=1 for an unbiased estimator.
        # This is strictly equivalent to your previous logic but statistically corrected for small M.
        diagonal_terms = np.var(mu_array, axis=1, ddof=1)
    
    diagonal_terms = np.clip(diagonal_terms, 0.0, None)
    eigenvalues = diagonal_terms + sigma_squared
    log_det = np.sum(np.log(eigenvalues), axis=1)

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
        diagonal_terms = np.mean((mu_array - reference_features) ** 2, axis=1)
    else:
        diagonal_terms = np.var(mu_array, axis=1, ddof=1)
        
    diagonal_terms = np.clip(diagonal_terms, 0.0, None)
    
    # Trace is simply the sum of all dimensional variances
    trace = np.sum(diagonal_terms, axis=1)

    if len(mu_array.shape) == 2:
        return trace[0]
    return trace


def distance_to_anchor(mu_array: np.ndarray) -> np.ndarray:
    """
    Calculate the Average Cosine Distance from the perturbed samples
    to the unperturbed base image (the anchor). 
    U_dist = 1/M * sum(1 - cosine_similarity(z_0, z_m))
    """
    mu_array = np.asarray(mu_array, dtype=np.float32)

    has_batch = len(mu_array.shape) == 3
    if not has_batch:
        mu_array = mu_array[np.newaxis, ...]

    B, M, D = mu_array.shape
    
    # Anchor (unperturbed) and perturbations
    z_0 = mu_array[:, 0:1, :]  # Shape: (B, 1, D)
    z_m = mu_array[:, 1:, :]   # Shape: (B, M-1, D)
    
    # L2 Normalization (add epsilon to prevent div by zero)
    z_0_norm = z_0 / (np.linalg.norm(z_0, axis=-1, keepdims=True) + 1e-12)
    z_m_norm = z_m / (np.linalg.norm(z_m, axis=-1, keepdims=True) + 1e-12)
    
    # Cosine Similarity and Distance
    cos_sim = np.sum(z_0_norm * z_m_norm, axis=-1)  # Shape: (B, M-1)
    cos_dist = 1.0 - cos_sim
    avg_dist = np.mean(cos_dist, axis=1)           # Shape: (B,)
    
    if not has_batch:
        return avg_dist[0]
    return avg_dist
