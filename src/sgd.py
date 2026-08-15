# Stochastic gradient descent approximates sparse ratings as
# R ≈ global_mean + U @ V.T. Each step updates one user and one movie factor
# from a sampled rating. The per-rating updates are accelerated with Numba.

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit


def _prepare_sparse_ratings(
    train_df,
    n_users,
    n_movies,
):
    """Prepare centered ratings, aligned coordinates, and user/movie degrees.

    Degrees are used to distribute regularization across per-rating updates.
    """
    user_idx = train_df["user_idx"].to_numpy(dtype=np.int64)
    movie_idx = train_df["movie_idx"].to_numpy(dtype=np.int64)
    ratings = train_df["rating"].to_numpy(dtype=np.float64)

    # Factor matrices model deviations from the global mean.
    global_mean = float(ratings.mean())

    ratings_sparse = sp.csr_matrix(
        (ratings - global_mean, (user_idx, movie_idx)),
        shape=(n_users, n_movies),
        dtype=np.float64,
    )

    # Rebuild coordinates in the same order as ratings_sparse.data.
    user_idx = np.repeat(
        np.arange(n_users, dtype=np.int64),
        np.diff(ratings_sparse.indptr),
    )
    movie_idx = ratings_sparse.indices.astype(np.int64, copy=False)
    ratings = ratings_sparse.data + global_mean
    user_degree = np.diff(ratings_sparse.indptr).astype(np.int64, copy=False)
    movie_degree = np.bincount(movie_idx, minlength=n_movies).astype(
        np.int64,
        copy=False,
    )

    return user_idx, movie_idx, ratings, global_mean, ratings_sparse, user_degree, movie_degree


@njit(cache=True)
def _sgd_steps_kernel(
    U,
    V,
    user_idx,
    movie_idx,
    ratings_sparse_data,
    user_degree,
    movie_degree,
    sampled_positions,
    eta,
    reg,
):
    """Apply sampled SGD updates to U and V in place.

    Regularization is divided by user/movie degree so its total contribution
    over an epoch does not depend on how often a factor row is sampled.
    """
    rank = U.shape[1]

    for step in range(sampled_positions.shape[0]):
        position = sampled_positions[step]
        u = user_idx[position]
        i = movie_idx[position]

        prediction = 0.0
        for factor in range(rank):
            prediction += U[u, factor] * V[i, factor]
        error = prediction - ratings_sparse_data[position]

        user_reg = reg / user_degree[u]
        movie_reg = reg / movie_degree[i]

        for factor in range(rank):
            # Both updates use the factor values from before this step.
            u_old = U[u, factor]
            v_old = V[i, factor]
            U[u, factor] = u_old - eta * (
                error * v_old + user_reg * u_old
            )
            V[i, factor] = v_old - eta * (
                error * u_old + movie_reg * v_old
            )


@njit(cache=True)
def _training_statistics_kernel(
    U,
    V,
    user_idx,
    movie_idx,
    ratings_sparse_data,
    ratings,
    global_mean,
    reg,
    clip_min,
    clip_max,
):
    """Compute objective and raw/clipped RMSE on the full training set."""
    squared_error = 0.0
    squared_error_clipped = 0.0
    rank = U.shape[1]

    for k in range(ratings_sparse_data.shape[0]):
        prediction_target = 0.0
        for factor in range(rank):
            prediction_target += (
                U[user_idx[k], factor] * V[movie_idx[k], factor]
            )

        residual = prediction_target - ratings_sparse_data[k]
        squared_error += residual * residual

        # Manual clipping is compatible with Numba's nopython mode.
        prediction = global_mean + prediction_target
        if prediction < clip_min:
            prediction = clip_min
        elif prediction > clip_max:
            prediction = clip_max

        clipped_residual = prediction - ratings[k]
        squared_error_clipped += clipped_residual * clipped_residual

    factor_squared_norm = 0.0
    for u in range(U.shape[0]):
        for factor in range(rank):
            factor_squared_norm += U[u, factor] * U[u, factor]
    for i in range(V.shape[0]):
        for factor in range(rank):
            factor_squared_norm += V[i, factor] * V[i, factor]

    objective = 0.5 * squared_error + 0.5 * reg * factor_squared_norm
    average_objective = objective / ratings_sparse_data.shape[0]
    train_rmse_raw = np.sqrt(squared_error / ratings_sparse_data.shape[0])
    train_rmse = np.sqrt(squared_error_clipped / ratings_sparse_data.shape[0])

    return objective, average_objective, train_rmse_raw, train_rmse


def stochastic_gradient_descent_numba(
    train_df,
    n_users,
    n_movies,
    rank=20,
    eta=1e-3,
    reg=0.1,
    n_iter=20,
    init_scale=0.01,
    random_state=42,
    metrics_fn=None,
    clip_min=0.5,
    clip_max=5.0,
):
    """Fit sparse matrix factors with Numba-accelerated SGD.

    Returns user/movie factors, the global mean, and one history row per epoch.
    The routine always runs exactly `n_iter` epochs.
    """
    user_idx, movie_idx, ratings, global_mean, ratings_sparse, user_degree, movie_degree = _prepare_sparse_ratings(train_df, n_users, n_movies)

    rng = np.random.default_rng(random_state)
    # Nonzero initialization avoids a zero-gradient state.
    U = np.ascontiguousarray(
        init_scale * rng.normal(size=(n_users, rank)),
        dtype=np.float64,
    )
    V = np.ascontiguousarray(
        init_scale * rng.normal(size=(n_movies, rank)),
        dtype=np.float64,
    )

    # Unobserved users or movies fall back to the global mean.
    U[user_degree == 0] = 0.0
    V[movie_degree == 0] = 0.0

    n_observations = ratings_sparse.data.shape[0]
    history = []

    for iteration in range(1, n_iter + 1):
        # Each epoch samples one position per observation, with replacement.
        sampled_positions = rng.integers(0, n_observations, size=n_observations, dtype=np.int64)
        _sgd_steps_kernel(U, V, user_idx, movie_idx, ratings_sparse.data, user_degree, movie_degree, sampled_positions, float(eta), float(reg))
        objective, average_objective, train_rmse_raw, train_rmse = _training_statistics_kernel(U, V, user_idx, movie_idx, ratings_sparse.data, ratings, float(global_mean), float(reg), float(clip_min), float(clip_max))

        row = {
            "iteration": iteration,
            "steps": iteration * n_observations,
            "rank": rank,
            "eta": eta,
            "reg": reg,
            "init_scale": init_scale,
            "objective": float(objective),
            "average_objective": float(average_objective),
            "train_rmse_raw": float(train_rmse_raw),
            "train_rmse": float(train_rmse),
        }

        if metrics_fn is not None:
            extra_metrics = metrics_fn(U, V, global_mean)
            if extra_metrics is not None:
                row.update(extra_metrics)

        history.append(row)

    return U, V, global_mean, pd.DataFrame(history)


# Short aliases used by the notebooks.
sgd_numba = stochastic_gradient_descent_numba
sgd = stochastic_gradient_descent_numba
