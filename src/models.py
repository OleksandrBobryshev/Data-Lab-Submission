import numpy as np
import pandas as pd
import scipy.sparse as sp

from evaluation import rmse_from_predictions
from evaluation import mae_from_predictions
from evaluation import evaluate_factor_model

def hard_threshold(M, rank):
    """Return the rank-limited approximation of a dense matrix using SVD."""
    # Compute the economy SVD of the matrix.
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    # Reconstruct the matrix from the first `rank` components.
    return (U[:, :rank] * s[:rank]) @ Vt[:rank, :]


def hard_thresholding_model(R_train, train_mask, rank=2, n_iter=60, metrics_fn=None):
    """Complete a dense rating matrix with iterative hard thresholding."""
    # Convert the observation mask to boolean values.
    mask = train_mask.astype(bool)

    # Use the mean observed rating as the initial prediction.
    global_mean = R_train[mask].mean()
    X = np.full(R_train.shape, global_mean, dtype=float)

    history = []

    for i in range(1, n_iter + 1):

        # Fill missing entries with the current predictions.
        Z = X.copy()
        Z[mask] = R_train[mask]

        # Keep only the selected number of singular components.
        X = hard_threshold(Z, rank)
        # Limit predictions to the valid rating range.
        X = np.clip(X, 0.5, 5.0)

        # Calculate training RMSE on observed ratings.
        train_preds = X[mask]
        train_true = R_train[mask]
        train_rmse = rmse_from_predictions(train_true, train_preds)

        row = {
            "iteration": i,
            "rank": rank,
            "train_rmse": train_rmse
        }

        # Calculate optional validation metrics.
        if metrics_fn is not None:
            extra_metrics = metrics_fn(X)
            row.update(extra_metrics)

        history.append(row)

    # Convert the iteration history to a DataFrame.
    history_df = pd.DataFrame(history)

    return X, history_df


def soft_threshold(M, tau):
    """Shrink the singular values of a dense matrix by tau."""
    # Compute the economy SVD of the matrix.
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    # Subtract `tau` and set negative singular values to zero.
    s_soft = np.maximum(s - tau, 0)

    # Reconstruct the matrix with the reduced singular values.
    return (U * s_soft) @ Vt


def soft_thresholding_model(R_train, train_mask, tau=20.0, n_iter=20, metrics_fn=None):
    """Complete a dense rating matrix with iterative soft thresholding."""
    # Convert the observation mask to boolean values.
    mask = train_mask.astype(bool)

    # Use the mean observed rating as the initial prediction.
    global_mean = R_train[mask].mean()
    X = np.full(R_train.shape, global_mean, dtype=float)

    history = []

    for i in range(1, n_iter + 1):
        # Fill missing entries with the current predictions.
        Z = X.copy()
        Z[mask] = R_train[mask]

        # Apply soft thresholding to the completed matrix.
        X = soft_threshold(Z, tau)
        # Limit predictions to the valid rating range.
        X = np.clip(X, 0.5, 5.0)

        # Calculate training RMSE on observed ratings.
        train_preds = X[mask]
        train_true = R_train[mask]
        train_rmse = rmse_from_predictions(train_true, train_preds)

        row = {
            "iteration": i,
            "tau": tau,
            "train_rmse": train_rmse
        }

        # Calculate optional validation metrics.
        if metrics_fn is not None:
            extra_metrics = metrics_fn(X)
            row.update(extra_metrics)

        history.append(row)

    # Convert the iteration history to a DataFrame.
    history_df = pd.DataFrame(history)

    return X, history_df