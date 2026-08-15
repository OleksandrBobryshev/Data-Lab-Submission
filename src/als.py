# Alternating Least Squares (ALS) approximates the sparse rating matrix as
# R ≈ global_mean + U @ V.T. It alternately fixes one factor matrix and solves
# regularized least-squares problems for the other. This file provides a fast
# Numba implementation and a NumPy/SciPy reference implementation.

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit, prange


def prepare_sparse_als_data(train_df, n_users, n_movies):
    """Prepare centered ratings and CSR/CSC views used by the ALS sweeps.

    The input must contain `user_idx`, `movie_idx`, and `rating`. The function
    also keeps flat coordinate arrays for efficient metric calculation.
    """
    user_idx = train_df["user_idx"].to_numpy(dtype=np.int64)
    movie_idx = train_df["movie_idx"].to_numpy(dtype=np.int64)
    ratings = train_df["rating"].to_numpy(dtype=np.float64)

    # Centering lets zero factors fall back to the global-mean prediction.
    global_mean = float(ratings.mean())
    ratings_centered = np.ascontiguousarray(ratings - global_mean)

    R_csr = sp.csr_matrix(
        (ratings_centered, (user_idx, movie_idx)),
        shape=(n_users, n_movies),
        dtype=np.float64,
    )
    # R_csr.sum_duplicates()

    # CSR is efficient by user; CSC is efficient by movie.
    R_csc = R_csr.tocsc()

    return user_idx, movie_idx, ratings, global_mean, R_csr, R_csc


@njit(cache=True)
def als_user_sweep_serial(csr_data, csr_indices, csr_indptr, V, reg, U):
    """Update all user factors with movie factors fixed; modifies U in place."""
    n_users = csr_indptr.shape[0] - 1
    rank = V.shape[1]

    for u in range(n_users):
        start = csr_indptr[u]
        end = csr_indptr[u + 1]

        # Users without ratings fall back to the global mean.
        if start == end:
            U[u, :] = 0.0
            continue

        A = np.zeros((rank, rank), dtype=np.float64)
        b = np.zeros(rank, dtype=np.float64)

        for pos in range(start, end):
            m = csr_indices[pos]
            y = csr_data[pos]

            for p in range(rank):
                v_p = V[m, p]
                b[p] += v_p * y

                for q in range(rank):
                    A[p, q] += v_p * V[m, q]

        # Ridge regularization keeps the system stable.
        for p in range(rank):
            A[p, p] += reg

        U[u, :] = np.linalg.solve(A, b)


@njit(cache=True)
def als_movie_sweep_serial(csc_data, csc_indices, csc_indptr, U, reg, V):
    """Update all movie factors with user factors fixed; modifies V in place."""
    n_movies = csc_indptr.shape[0] - 1
    rank = U.shape[1]

    for m in range(n_movies):
        start = csc_indptr[m]
        end = csc_indptr[m + 1]

        # Movies without ratings fall back to the global mean.
        if start == end:
            V[m, :] = 0.0
            continue

        A = np.zeros((rank, rank), dtype=np.float64)
        b = np.zeros(rank, dtype=np.float64)

        for pos in range(start, end):
            u = csc_indices[pos]
            y = csc_data[pos]

            for p in range(rank):
                u_p = U[u, p]
                b[p] += u_p * y

                for q in range(rank):
                    A[p, q] += u_p * U[u, q]

        for p in range(rank):
            A[p, p] += reg

        V[m, :] = np.linalg.solve(A, b)


@njit(parallel=True, cache=True)
def als_user_sweep_parallel(csr_data, csr_indices, csr_indptr, V, reg, U):
    """Parallel user-factor update; each user is solved independently."""
    n_users = csr_indptr.shape[0] - 1
    rank = V.shape[1]

    for u in prange(n_users):
        start = csr_indptr[u]
        end = csr_indptr[u + 1]

        if start == end:
            U[u, :] = 0.0
            continue

        A = np.zeros((rank, rank), dtype=np.float64)
        b = np.zeros(rank, dtype=np.float64)

        for pos in range(start, end):
            m = csr_indices[pos]
            y = csr_data[pos]

            for p in range(rank):
                v_p = V[m, p]
                b[p] += v_p * y

                for q in range(rank):
                    A[p, q] += v_p * V[m, q]

        for p in range(rank):
            A[p, p] += reg

        U[u, :] = np.linalg.solve(A, b)


@njit(parallel=True, cache=True)
def als_movie_sweep_parallel(csc_data, csc_indices, csc_indptr, U, reg, V):
    """Parallel movie-factor update; each movie is solved independently."""
    n_movies = csc_indptr.shape[0] - 1
    rank = U.shape[1]

    for m in prange(n_movies):
        start = csc_indptr[m]
        end = csc_indptr[m + 1]

        if start == end:
            V[m, :] = 0.0
            continue

        A = np.zeros((rank, rank), dtype=np.float64)
        b = np.zeros(rank, dtype=np.float64)

        for pos in range(start, end):
            u = csc_indices[pos]
            y = csc_data[pos]

            for p in range(rank):
                u_p = U[u, p]
                b[p] += u_p * y

                for q in range(rank):
                    A[p, q] += u_p * U[u, q]

        for p in range(rank):
            A[p, p] += reg

        V[m, :] = np.linalg.solve(A, b)


@njit(cache=True)
def predict_pairs_numba_kernel(U, V, user_idx, movie_idx, global_mean):
    """Predict ratings for observed user-movie pairs without clipping."""
    n = user_idx.shape[0]
    predictions = np.empty(n, dtype=np.float64)

    for k in range(n):
        predictions[k] = global_mean + np.dot(U[user_idx[k]], V[movie_idx[k]])

    return predictions


@njit(cache=True)
def training_statistics_numba_kernel(
    U,
    V,
    user_idx,
    movie_idx,
    ratings,
    global_mean,
    reg,
    clip_min,
    clip_max,
):
    """Compute the regularized ALS objective and raw/clipped training RMSE."""
    n = ratings.shape[0]
    squared_error_raw = 0.0
    squared_error_clipped = 0.0

    for k in range(n):
        prediction = global_mean + np.dot(U[user_idx[k]], V[movie_idx[k]])

        residual_raw = prediction - ratings[k]
        squared_error_raw += residual_raw * residual_raw

        # Manual clipping is compatible with Numba's nopython mode.
        prediction_clipped = prediction
        if prediction_clipped < clip_min:
            prediction_clipped = clip_min
        elif prediction_clipped > clip_max:
            prediction_clipped = clip_max

        residual_clipped = prediction_clipped - ratings[k]
        squared_error_clipped += residual_clipped * residual_clipped

    factor_squared_norm = (
        np.dot(U.ravel(), U.ravel())
        + np.dot(V.ravel(), V.ravel())
    )

    objective = 0.5 * squared_error_raw + 0.5 * reg * factor_squared_norm
    rmse_raw = np.sqrt(squared_error_raw / n)
    rmse_clipped = np.sqrt(squared_error_clipped / n)

    return objective, rmse_raw, rmse_clipped


def als_model_numba(
    train_df,
    n_users,
    n_movies,
    rank=20,
    reg=0.1,
    n_iter=30,
    random_state=42,
    init_scale=0.01,
    metrics_fn=None,
    parallel=True,
    clip_min=0.5,
    clip_max=5.0,
    min_iter=5,
    objective_tol=None,
    patience=3,
):
    """Fit ALS with Numba-compiled serial or parallel update kernels.

    Returns user and movie factors, the global mean, and iteration history.
    Early stopping is optional and based on relative objective improvement.
    """
    user_idx, movie_idx, ratings, global_mean, R_csr, R_csc = (
        prepare_sparse_als_data(train_df, n_users, n_movies)
    )

    rng = np.random.default_rng(random_state)
    U = np.zeros((n_users, rank), dtype=np.float64)
    # Random movie factors break symmetry before the first user update.
    V = np.ascontiguousarray(
        init_scale * rng.normal(size=(n_movies, rank)),
        dtype=np.float64,
    )

    if parallel:
        user_sweep = als_user_sweep_parallel
        movie_sweep = als_movie_sweep_parallel
    else:
        user_sweep = als_user_sweep_serial
        movie_sweep = als_movie_sweep_serial

    history = []
    previous_objective = None
    small_decrease_count = 0

    for iteration in range(1, n_iter + 1):

        # Update user factors with movies fixed.
        user_sweep(
            R_csr.data,
            R_csr.indices,
            R_csr.indptr,
            V,
            float(reg),
            U,
        )

        # Update movie factors with users fixed.
        movie_sweep(
            R_csc.data,
            R_csc.indices,
            R_csc.indptr,
            U,
            float(reg),
            V,
        )

        objective, train_rmse_raw, train_rmse = training_statistics_numba_kernel(
            U,
            V,
            user_idx,
            movie_idx,
            ratings,
            float(global_mean),
            float(reg),
            float(clip_min),
            float(clip_max),
        )

        row = {
            "iteration": iteration,
            "rank": rank,
            "reg": reg,
            "objective": float(objective),
            "train_rmse_raw": float(train_rmse_raw),
            "train_rmse": float(train_rmse),
        }

        if previous_objective is None:
            relative_decrease = np.nan
        else:
            relative_decrease = (previous_objective - objective) / max(1.0, previous_objective)

        row["relative_objective_decrease"] = relative_decrease

        if metrics_fn is not None:
            extra_metrics = metrics_fn(
                U,
                V,
                global_mean,
            )
            row.update(extra_metrics)

        history.append(row)

        # Count consecutive iterations with negligible objective improvement.
        if (
            objective_tol is not None
            and previous_objective is not None
            and iteration >= min_iter
            and 0.0 <= relative_decrease <= objective_tol
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


# NumPy/SciPy reference implementation used to validate the Numba version.


def training_statistics_sparse(
    U,
    V,
    user_idx,
    movie_idx,
    ratings,
    global_mean,
    reg,
    clip_min,
    clip_max,
    batch_size=200_000,
):
    """Compute objective and RMSE with NumPy in memory-limited batches."""
    squared_error_raw = 0.0
    squared_error_clipped = 0.0

    for start in range(0, ratings.shape[0], batch_size):
        end = start + batch_size
        batch_users = user_idx[start:end]
        batch_movies = movie_idx[start:end]
        batch_ratings = ratings[start:end]

        predictions = global_mean + np.sum(
            U[batch_users] * V[batch_movies],
            axis=1,
        )
        residuals_raw = predictions - batch_ratings
        squared_error_raw += np.dot(residuals_raw, residuals_raw)

        predictions_clipped = np.clip(predictions, clip_min, clip_max)
        residuals_clipped = predictions_clipped - batch_ratings
        squared_error_clipped += np.dot(residuals_clipped, residuals_clipped)

    factor_squared_norm = np.dot(U.ravel(), U.ravel()) + np.dot(V.ravel(), V.ravel())

    objective = 0.5 * squared_error_raw + 0.5 * reg * factor_squared_norm
    rmse_raw = np.sqrt(squared_error_raw / ratings.shape[0])
    rmse_clipped = np.sqrt(squared_error_clipped / ratings.shape[0])

    return objective, rmse_raw, rmse_clipped


def als_model_sparse(
    train_df,
    n_users,
    n_movies,
    rank=20,
    reg=0.1,
    n_iter=30,
    random_state=42,
    init_scale=0.01,
    metrics_fn=None,
    clip_min=0.5,
    clip_max=5.0,
    min_iter=5,
    objective_tol=None,
    patience=3,
):
    """Fit the same ALS model using readable NumPy/SciPy operations.

    This reference version is used to validate the faster Numba implementation.
    It returns user/movie factors, the global mean, and iteration history.
    """
    user_idx, movie_idx, ratings, global_mean, R_csr, R_csc = (
        prepare_sparse_als_data(train_df, n_users, n_movies)
    )

    rng = np.random.default_rng(random_state)
    U = np.zeros((n_users, rank), dtype=np.float64)
    V = init_scale * rng.normal(size=(n_movies, rank))

    reg_identity = reg * np.eye(rank, dtype=np.float64)
    history = []
    previous_objective = None
    small_decrease_count = 0

    for iteration in range(1, n_iter + 1):
        # Update user factors with movies fixed.
        for u in range(n_users):
            start = R_csr.indptr[u]
            end = R_csr.indptr[u + 1]

            if start == end:
                U[u, :] = 0.0
                continue

            movie_ids = R_csr.indices[start:end]
            ratings_u = R_csr.data[start:end]
            V_u = V[movie_ids]

            A = V_u.T @ V_u + reg_identity
            b = V_u.T @ ratings_u
            U[u, :] = np.linalg.solve(A, b)

        # Update movie factors with users fixed.
        for m in range(n_movies):
            start = R_csc.indptr[m]
            end = R_csc.indptr[m + 1]

            if start == end:
                V[m, :] = 0.0
                continue

            user_ids = R_csc.indices[start:end]
            ratings_m = R_csc.data[start:end]
            U_m = U[user_ids]

            A = U_m.T @ U_m + reg_identity
            b = U_m.T @ ratings_m
            V[m, :] = np.linalg.solve(A, b)

        objective, train_rmse_raw, train_rmse = training_statistics_sparse(
            U,
            V,
            user_idx,
            movie_idx,
            ratings,
            global_mean,
            reg,
            clip_min,
            clip_max,
        )

        row = {
            "iteration": iteration,
            "rank": rank,
            "reg": reg,
            "objective": float(objective),
            "train_rmse_raw": float(train_rmse_raw),
            "train_rmse": float(train_rmse),
        }

        if previous_objective is None:
            relative_decrease = np.nan
        else:
            relative_decrease = (previous_objective - objective) / max(1.0, previous_objective)

        row["relative_objective_decrease"] = relative_decrease

        if metrics_fn is not None:
            extra_metrics = metrics_fn(
                U,
                V,
                global_mean,
            )
            row.update(extra_metrics)

        history.append(row)

        if (
            objective_tol is not None
            and previous_objective is not None
            and iteration >= min_iter
            and 0.0 <= relative_decrease <= objective_tol
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

