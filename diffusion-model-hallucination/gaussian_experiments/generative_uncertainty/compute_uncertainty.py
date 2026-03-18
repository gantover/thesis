import jax.numpy as jnp

def get_uncertainty_scores(
    uncertainty_ensemble,
    kind: str = "diagonal_gaussian_entropy",
    eps: float = 1e-8,
):

    if kind == "diagonal_gaussian_entropy":
        # reproduction of Jazbec et al. method from appendix B.1
        # gaussian obtained with moment matching, followed by diagnonal covariance assumption
        variances = jnp.var(uncertainty_ensemble, axis=0)
        uncertainty_scores = 0.5 * jnp.sum(jnp.log(variances + eps), axis=1)

    elif kind == "full_gaussian_entropy":
        # removing the diagonal covariance assumption
        uncertainty_scores = []
        for i in range(len(uncertainty_ensemble[0])): # Iterate over each sample
            points = uncertainty_ensemble[:, i, :] # Shape: (num_models, 2)

            # the full 2x2 covariance matrix
            cov_matrix = jnp.cov(points, rowvar=False) 
            
            # Calculate the log determinant (adding a small epsilon to the diagonal for stability)
            cov_matrix += jnp.eye(2) * eps
            sign, logdet = jnp.linalg.slogdet(cov_matrix)
            
            # Entropy is proportional to log determinant
            uncertainty_scores.append(0.5 * logdet)
            
        uncertainty_scores = jnp.array(uncertainty_scores)

    elif kind == "raw_variance":
        uncertainty_scores = jnp.sum(jnp.var(uncertainty_ensemble, axis=0), axis=1)

    else:
        raise ValueError(f"Unknown uncertainty calculation method: {kind}")

    return uncertainty_scores