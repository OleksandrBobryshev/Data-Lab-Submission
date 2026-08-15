# Full-batch gradient descent approximates sparse ratings as
# R ≈ global_mean + U @ V.T. Each iteration computes gradients from all
# observed ratings and updates both factor matrices simultaneously.

import numpy as np
import pandas as pd
import scipy.sparse as sp

def _prepare_sparse_ratings(train_df, n_users, n_movies):
    """Prepare centered sparse ratings and aligned coordinate arrays."""

    if train_df.duplicated(["user_idx", "movie_idx"]).any():
        raise ValueError("train_df must contain at most one rating per user-movie pair")

    user_idx = train_df["user_idx"].to_numpy(dtype=np.int64)
    movie_idx = train_df["movie_idx"].to_numpy(dtype=np.int64)
    ratings = train_df["rating"].to_numpy(dtype=np.float64)
    global_mean = float(ratings.mean())

    ratings_centered = sp.csr_matrix(
        (ratings - global_mean, (user_idx, movie_idx)),
        shape=(n_users, n_movies),
        dtype=np.float64,
    )

    # Rebuild coordinates in the same order as ratings_centered.data.
    user_idx = np.repeat(
        np.arange(n_users, dtype=np.int64),
        np.diff(ratings_centered.indptr),
    )
    movie_idx = ratings_centered.indices
    ratings = ratings_centered.data + global_mean

    return user_idx, movie_idx, ratings, global_mean, ratings_centered


def _observed_predictions(U, V, user_idx, movie_idx):
    """Compute centered predictions only for observed user-movie pairs."""
    return np.sum(U[user_idx] * V[movie_idx], axis=1)


def matrix_factorization_objective(
    U,
    V,
    ratings_centered,
    user_idx,
    movie_idx,
    reg=0.0,
):
    """Evaluate squared error plus L2 regularization."""

    predictions = _observed_predictions(U, V, user_idx, movie_idx)
    errors = predictions - ratings_centered.data
    data_fit = 0.5 * np.dot(errors, errors)
    penalty = 0.5 * reg * (np.sum(U * U) + np.sum(V * V))
    return float(data_fit + penalty)


def matrix_factorization_gradients(
    U,
    V,
    ratings_centered,
    user_idx,
    movie_idx,
    reg=0.0,
):
    """Compute full-batch gradients for both factor matrices."""

    predictions = _observed_predictions(U, V, user_idx, movie_idx)
    errors = predictions - ratings_centered.data

    residual = ratings_centered.copy()
    residual.data = errors
    grad_U = residual @ V + reg * U
    grad_V = residual.T @ U + reg * V
    return grad_U, grad_V

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
    min_iter=5,
    objective_tol=None,
    patience=3,
):
    """Fit sparse matrix factors with full-batch gradient descent.

    Returns user/movie factors, the global mean, and iteration history.
    Optional early stopping uses relative objective change.
    """
    user_idx, movie_idx, ratings, global_mean, ratings_centered = (
        _prepare_sparse_ratings(
            train_df,
            int(n_users),
            int(n_movies),
        )
    )

    rng = np.random.default_rng(random_state)
    # Nonzero initialization avoids a zero-gradient state.
    U = init_scale * rng.normal(size=(n_users, rank))
    V = init_scale * rng.normal(size=(n_movies, rank))

    history = []
    previous_objective = None
    small_decrease_count = 0

    for iteration in range(1, n_iter + 1):
        # Compute both gradients before overwriting either factor.
        grad_U, grad_V = matrix_factorization_gradients(
            U,
            V,
            ratings_centered,
            user_idx,
            movie_idx,
            reg=reg,
        )
        U_next = U - eta * grad_U
        V_next = V - eta * grad_V

        # A fixed learning rate that is too large can make the factors diverge.
        if not np.all(np.isfinite(U_next)) or not np.all(np.isfinite(V_next)):
            raise FloatingPointError(
                "non-finite factors encountered; the fixed step size may be too large"
            )

        U, V = U_next, V_next

        predictions_centered = _observed_predictions(
            U,
            V,
            user_idx,
            movie_idx,
        )
        residuals = predictions_centered - ratings_centered.data
        objective = matrix_factorization_objective(
            U,
            V,
            ratings_centered,
            user_idx,
            movie_idx,
            reg=reg,
        )
        predictions = global_mean + predictions_centered
        train_rmse_raw = float(np.sqrt(np.mean(residuals * residuals)))
        clipped_predictions = np.clip(predictions, clip_min, clip_max)
        train_rmse = float(
            np.sqrt(np.mean((clipped_predictions - ratings) ** 2))
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

        if previous_objective is None:
            relative_decrease = np.nan
        else:
            relative_decrease = abs(previous_objective - objective) / max(
                1.0,
                previous_objective,
            )

        row["relative_objective_decrease"] = relative_decrease

        if metrics_fn is not None:
            extra_metrics = metrics_fn(U, V, global_mean)
            if extra_metrics is not None:
                row.update(extra_metrics)

        history.append(row)

        # Count consecutive iterations with negligible objective change.
        if (
            objective_tol is not None
            and previous_objective is not None
            and iteration >= min_iter
            and relative_decrease <= objective_tol
        ):
            small_decrease_count += 1
        else:
            small_decrease_count = 0

        previous_objective = objective

        if (
            objective_tol is not None
            and small_decrease_count >= patience
        ):
            break

    return U, V, global_mean, pd.DataFrame(history)
