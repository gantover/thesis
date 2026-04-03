import numpy as np
import jax.numpy as jnp


def _sanitize_ensemble(uncertainty_ensemble):
    """Return float64 array with non-finite values replaced by NaN."""
    arr = np.asarray(uncertainty_ensemble, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected uncertainty_ensemble shape (num_models, num_samples, dim), got {arr.shape}."
        )
    return np.where(np.isfinite(arr), arr, np.nan)

def get_uncertainty_scores(
    uncertainty_ensemble,
    kind: str = "diagonal_gaussian_entropy",
    eps: float = 1e-8,
    var_clip_max: float = 1e8,
):
    """Compute uncertainty scores robustly, even with occasional non-finite ensemble samples.

    Non-finite values are ignored in moment estimates. If too few finite ensemble members
    exist for a sample, the score is mapped to a large but finite value.
    """
    if var_clip_max <= eps:
        raise ValueError(f"var_clip_max must be > eps. Got var_clip_max={var_clip_max}, eps={eps}.")

    ensemble = _sanitize_ensemble(uncertainty_ensemble)
    num_models, num_samples, dim = ensemble.shape
    low_score = 0.5 * dim * np.log(eps)
    high_score = 0.5 * dim * np.log(var_clip_max)

    if kind == "diagonal_gaussian_entropy":
        # reproduction of Jazbec et al. method from appendix B.1
        # gaussian obtained with moment matching, followed by diagonal covariance assumption
        variances = np.nanvar(ensemble, axis=0)
        variances = np.nan_to_num(variances, nan=var_clip_max, posinf=var_clip_max, neginf=eps)
        variances = np.clip(variances, eps, var_clip_max)
        uncertainty_scores = 0.5 * np.sum(np.log(variances), axis=1)

    elif kind == "full_gaussian_entropy":
        # removing the diagonal covariance assumption
        uncertainty_scores = np.empty(num_samples, dtype=np.float64)
        for i in range(num_samples):
            points = ensemble[:, i, :]  # shape: (num_models, dim)
            finite_rows = np.isfinite(points).all(axis=1)
            points = points[finite_rows]

            if points.shape[0] < 2:
                uncertainty_scores[i] = high_score
                continue

            cov_matrix = np.cov(points, rowvar=False)
            cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=var_clip_max, neginf=0.0)
            cov_matrix += np.eye(dim) * eps
            
            sign, logdet = np.linalg.slogdet(cov_matrix)
            if sign <= 0 or not np.isfinite(logdet):
                uncertainty_scores[i] = high_score
            else:
                uncertainty_scores[i] = 0.5 * logdet

    elif kind == "raw_variance":
        variances = np.nanvar(ensemble, axis=0)
        variances = np.nan_to_num(variances, nan=var_clip_max, posinf=var_clip_max, neginf=eps)
        variances = np.clip(variances, eps, var_clip_max)
        uncertainty_scores = np.sum(variances, axis=1)

    else:
        raise ValueError(f"Unknown uncertainty calculation method: {kind}")

    uncertainty_scores = np.nan_to_num(
        uncertainty_scores,
        nan=high_score,
        posinf=high_score,
        neginf=low_score,
    )
    return jnp.asarray(uncertainty_scores)