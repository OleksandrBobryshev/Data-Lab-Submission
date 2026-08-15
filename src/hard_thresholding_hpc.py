"""Matrix-free hard thresholding for sparse rating data.

Observed ratings replace the current predictions at every iteration, after
which a truncated SVD restores the selected rank without forming a dense matrix.
"""

from time import perf_counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, svds


@njit(cache=True, parallel=True)
def predict_pairs(U, singular_values, V, user_idx, movie_idx):
    """Predict selected user-movie pairs without forming the full matrix."""
    predictions = np.empty(user_idx.shape[0], dtype=np.float64)

    for position in prange(user_idx.shape[0]):
        user = user_idx[position]
        movie = movie_idx[position]
        prediction = 0.0

        for factor in range(singular_values.shape[0]):
            prediction += (
                U[user, factor]
                * singular_values[factor]
                * V[movie, factor]
            )

        predictions[position] = prediction

    return predictions


def _prepare_ratings(train_df, n_users, n_movies):
    """Convert ratings to CSR format and align the coordinate arrays."""
    user_idx = train_df["user_idx"].to_numpy(dtype=np.int64)
    movie_idx = train_df["movie_idx"].to_numpy(dtype=np.int64)
    ratings = train_df["rating"].to_numpy(dtype=np.float64)

    observed = sp.csr_matrix(
        (ratings, (user_idx, movie_idx)),
        shape=(n_users, n_movies),
        dtype=np.float64,
    )
    observed.sort_indices()

    # Rebuild coordinates in the same order as observed.data.
    user_idx = np.repeat(
        np.arange(n_users, dtype=np.int64),
        np.diff(observed.indptr),
    )
    movie_idx = observed.indices.astype(np.int64, copy=False)
    ratings = observed.data.copy()

    return user_idx, movie_idx, ratings, observed


def hard_thresholding_model_matrix_free(
    train_df,
    n_users,
    n_movies,
    rank=10,
    n_iter=15,
    metrics_fn=None,
    svd_tol=1e-4,
    svd_maxiter=300,
    random_state=42,
):
    """Fit iterative hard thresholding without a dense user-movie matrix.

    Returns the truncated SVD factors and an iteration history. A LinearOperator
    represents the imputed matrix during each partial-SVD update.
    """

    n_users = int(n_users)
    n_movies = int(n_movies)
    rank = int(rank)
    n_iter = int(n_iter)

    if n_users <= 0 or n_movies <= 0:
        raise ValueError("n_users and n_movies must be positive")
    if not 1 <= rank < min(n_users, n_movies):
        raise ValueError("rank must be between 1 and min(n_users, n_movies) - 1")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")

    user_idx, movie_idx, ratings, observed = _prepare_ratings(
        train_df,
        n_users,
        n_movies,
    )

    global_mean = float(ratings.mean())
    U = np.ones((n_users, 1), dtype=np.float64) / np.sqrt(n_users)
    singular_values = np.array(
        [global_mean * np.sqrt(n_users * n_movies)],
        dtype=np.float64,
    )
    V = np.ones((n_movies, 1), dtype=np.float64) / np.sqrt(n_movies)

    correction = observed.copy()
    history = []

    for iteration in range(1, n_iter + 1):
        iteration_start = perf_counter()

        train_predictions = predict_pairs(
            U,
            singular_values,
            V,
            user_idx,
            movie_idx,
        )
        correction.data[:] = ratings - train_predictions
        correction_transpose = correction.T

        def matvec(vector):
            """Multiply the virtual imputed matrix by a vector."""
            vector = np.asarray(vector).reshape(-1)
            return (
                U @ (singular_values * (V.T @ vector))
                + correction @ vector
            )

        def rmatvec(vector):
            """Multiply the transpose of the virtual matrix by a vector."""
            vector = np.asarray(vector).reshape(-1)
            return (
                V @ (singular_values * (U.T @ vector))
                + correction_transpose @ vector
            )

        def matmat(matrix):
            """Multiply the virtual imputed matrix by a dense matrix."""
            return (
                U @ (singular_values[:, None] * (V.T @ matrix))
                + correction @ matrix
            )

        def rmatmat(matrix):
            """Multiply the transpose of the virtual matrix by a dense matrix."""
            return (
                V @ (singular_values[:, None] * (U.T @ matrix))
                + correction_transpose @ matrix
            )

        operator = LinearOperator(
            shape=(n_users, n_movies),
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            dtype=np.float64,
        )

        starting_vector = V[:, 0] if n_users > n_movies else U[:, 0]
        svd_start = perf_counter()

        # ARPACK computes only the leading components of the virtual matrix.
        U_new, singular_values_new, Vt_new = svds(
            operator,
            k=rank,
            which="LM",
            v0=starting_vector,
            tol=svd_tol,
            maxiter=svd_maxiter,
            solver="arpack",
            random_state=random_state,
        )
        svd_seconds = perf_counter() - svd_start

        order = np.argsort(singular_values_new)[::-1]
        U = U_new[:, order]
        singular_values = singular_values_new[order]
        V = Vt_new[order, :].T

        train_predictions = predict_pairs(
            U,
            singular_values,
            V,
            user_idx,
            movie_idx,
        )
        train_predictions = np.clip(train_predictions, 0.5, 5.0)
        train_rmse = float(np.sqrt(np.mean((ratings - train_predictions) ** 2)))

        row = {
            "iteration": iteration,
            "rank": rank,
            "train_rmse": train_rmse,
            "svd_seconds": svd_seconds,
            "iteration_seconds": perf_counter() - iteration_start,
        }
        if metrics_fn is not None:
            row.update(metrics_fn(U, singular_values, V))
        history.append(row)

    return U, singular_values, V, pd.DataFrame(history)
