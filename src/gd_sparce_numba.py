# Full-batch gradient descent approximates sparse ratings as
# R ≈ global_mean + U @ V.T. At each iteration it computes the exact gradient
# over all observed ratings and updates both factor matrices simultaneously.
# The expensive per-rating operations are accelerated with Numba.

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit


def _prepare_sparse_ratings(train_df, n_users, n_movies):
    """Prepare centered sparse ratings and aligned coordinate arrays.

    The input must contain `user_idx`, `movie_idx`, and `rating`. Coordinates
    are reordered to match the internal CSR data order used by the kernels.
    """
    # Duplicate coordinates would be summed by the sparse-matrix constructor.
    if train_df.duplicated(["user_idx", "movie_idx"]).any():
        raise ValueError("train_df must contain at most one rating per user-movie pair")

    user_idx = train_df["user_idx"].to_numpy(dtype=np.int64)
    movie_idx = train_df["movie_idx"].to_numpy(dtype=np.int64)
    ratings = train_df["rating"].to_numpy(dtype=np.float64)
    # Factor matrices model deviations from the global mean.
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
    movie_idx = ratings_centered.indices.astype(np.int64, copy=False)
    ratings = ratings_centered.data + global_mean

    return user_idx, movie_idx, ratings, global_mean, ratings_centered


@njit(cache=True)
def _observed_predictions_kernel(U, V, user_idx, movie_idx):
    """Compute centered predictions only for observed user-movie pairs."""
    n_observations = user_idx.shape[0]
    rank = U.shape[1]
    predictions = np.empty(n_observations, dtype=np.float64)

    for k in range(n_observations):
        u = user_idx[k]
        i = movie_idx[k]
        prediction = 0.0

        for factor in range(rank):
            prediction += U[u, factor] * V[i, factor]

        predictions[k] = prediction

    return predictions


def _observed_predictions(U, V, user_idx, movie_idx):
    """Call the compiled prediction kernel."""
    return _observed_predictions_kernel(U, V, user_idx, movie_idx)


@njit(cache=True)
def _objective_kernel(
    U,
    V,
    ratings_centered_data,
    user_idx,
    movie_idx,
    reg,
):
    """Evaluate squared error plus L2 regularization."""
    n_observations = user_idx.shape[0]
    rank = U.shape[1]
    squared_error = 0.0

    for k in range(n_observations):
        u = user_idx[k]
        i = movie_idx[k]
        prediction = 0.0

        for factor in range(rank):
            prediction += U[u, factor] * V[i, factor]

        error = prediction - ratings_centered_data[k]
        squared_error += error * error

    factor_squared_norm = 0.0

    for u in range(U.shape[0]):
        for factor in range(rank):
            factor_squared_norm += U[u, factor] * U[u, factor]

    for i in range(V.shape[0]):
        for factor in range(rank):
            factor_squared_norm += V[i, factor] * V[i, factor]

    return 0.5 * squared_error + 0.5 * reg * factor_squared_norm


@njit(cache=True)
def _gradients_kernel(
    U,
    V,
    ratings_centered_data,
    user_idx,
    movie_idx,
    reg,
):
    """Compute full-batch gradients for both factor matrices."""
    rank = U.shape[1]
    grad_U = reg * U
    grad_V = reg * V

    for k in range(user_idx.shape[0]):
        u = user_idx[k]
        i = movie_idx[k]
        prediction = 0.0

        for factor in range(rank):
            prediction += U[u, factor] * V[i, factor]

        error = prediction - ratings_centered_data[k]

        for factor in range(rank):
            grad_U[u, factor] += error * V[i, factor]
            grad_V[i, factor] += error * U[u, factor]

    return grad_U, grad_V


def gradient_descent_numba(
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
    """Fit sparse matrix factors with Numba-accelerated full-batch GD.

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
    # Both factors must be nonzero initially to avoid a zero-gradient state.
    U = init_scale * rng.normal(size=(n_users, rank))
    V = init_scale * rng.normal(size=(n_movies, rank))

    history = []
    previous_objective = None
    small_decrease_count = 0

    for iteration in range(1, n_iter + 1):
        # Both gradients are evaluated before either factor matrix is updated.
        grad_U, grad_V = _gradients_kernel(
            U,
            V,
            ratings_centered.data,
            user_idx,
            movie_idx,
            float(reg),
        )
        U_next = U - eta * grad_U
        V_next = V - eta * grad_V

        # Stop immediately if the fixed learning rate causes divergence.
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
        objective = float(
            _objective_kernel(
                U,
                V,
                ratings_centered.data,
                user_idx,
                movie_idx,
                float(reg),
            )
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
