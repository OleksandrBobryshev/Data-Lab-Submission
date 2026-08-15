import numpy as np
import pandas as pd


def matrix_factorization_objective(U, V, ratings, observed_mask, reg=0.0):
    """Return the summed ridge-regularized matrix-factorization objective."""
    residual = (U @ V.T - ratings) * observed_mask # = P_omega(UV^T - R)
    data_fit = 0.5 * np.sum(residual * residual)
    penalty = 0.5 * reg * (np.sum(U * U) + np.sum(V * V))
    return float(data_fit + penalty)


def matrix_factorization_gradients(U, V, ratings, observed_mask, reg=0.0):
    """Return both full-batch gradients evaluated at the same ``(U, V)``."""
    residual = (U @ V.T - ratings) * observed_mask
    grad_U = residual @ V + reg * U
    grad_V = residual.T @ U + reg * V
    return grad_U, grad_V


def _prepare_dense_ratings(train_df, n_users, n_movies):

    if train_df.duplicated(["user_idx", "movie_idx"]).any():
        raise ValueError("train_df must contain at most one rating per user-movie pair")

    user_idx = train_df["user_idx"].to_numpy(dtype=np.int64)
    movie_idx = train_df["movie_idx"].to_numpy(dtype=np.int64)
    values = train_df["rating"].to_numpy(dtype=np.float64)
    ratings = np.zeros((n_users, n_movies), dtype=np.float64)
    observed_mask = np.zeros((n_users, n_movies), dtype=bool)
    ratings[user_idx, movie_idx] = values
    observed_mask[user_idx, movie_idx] = True
    return ratings, observed_mask


def gradient_descent(
    train_df,
    n_users,
    n_movies,
    rank=20,
    eta=1e-4,
    reg=0.0,
    n_iter=100,
    init_scale=0.1,
    random_state=42,
    metrics_fn=None,
    clip_min=0.5,
    clip_max=5.0,
):
    """
    Returns
    -------
    U, V : numpy.ndarray
        Learned factor matrices with shapes ``(n_users, rank)`` and
        ``(n_movies, rank)``.
    history : pandas.DataFrame
        Objective and training metrics after each completed update.
    """

    ratings, observed_mask = _prepare_dense_ratings(
        train_df,
        int(n_users),
        int(n_movies),
    )

    observed_ratings = ratings[observed_mask] # returns 1D array

    rng = np.random.default_rng(random_state)
    U = init_scale * rng.normal(size=(n_users, rank))
    V = init_scale * rng.normal(size=(n_movies, rank))

    history = []

    for iteration in range(1, n_iter + 1):
        # Both gradients are computed before either factor is overwritten.
        grad_U, grad_V = matrix_factorization_gradients(
            U,
            V,
            ratings,
            observed_mask,
            reg=reg,
        )
        U_next = U - eta * grad_U
        V_next = V - eta * grad_V

        if not np.all(np.isfinite(U_next)) or not np.all(np.isfinite(V_next)):
            raise FloatingPointError(
                "non-finite factors encountered; the fixed step size may be too large"
            )

        U, V = U_next, V_next

        objective = matrix_factorization_objective(
            U,
            V,
            ratings,
            observed_mask,
            reg=reg,
        )
        predictions = (U @ V.T)[observed_mask]
        residuals = predictions - observed_ratings
        train_rmse_raw = float(np.sqrt(np.mean(residuals * residuals)))
        clipped_predictions = np.clip(predictions, clip_min, clip_max)
        train_rmse = float(
            np.sqrt(np.mean((clipped_predictions - observed_ratings) ** 2))
        )

        row = {
            "iteration": iteration,
            "rank": rank,
            "eta": eta,
            "reg": reg,
            "init_scale": init_scale,
            "objective": objective,
            "train_rmse_raw": train_rmse_raw,
            "train_rmse": train_rmse,
        }

        if metrics_fn is not None:
            extra_metrics = metrics_fn(U, V)
            if extra_metrics is not None:
                row.update(extra_metrics)

        history.append(row)

    return U, V, pd.DataFrame(history)