import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse_from_predictions(y_true, y_pred):
    """
    Compute RMSE between true ratings and predicted ratings.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_from_predictions(y_true, y_pred):
    """
    Compute MAE between true ratings and predicted ratings.
    """
    return mean_absolute_error(y_true, y_pred)


def evaluate_matrix_model(X_hat, test_df):
    # For each row in test_df, take the predicted rating
    # at the corresponding user_idx and movie_idx.
    preds = X_hat[test_df["user_idx"].values, test_df["movie_idx"].values]
    y_true = test_df["rating"].values

    # True ratings are taken from the same rows of test_df,
    # so the order matches preds.
    rmse = rmse_from_predictions(y_true, preds)
    mae = mae_from_predictions(y_true, preds)

    return rmse, mae

#this function is used in the gradient descent algorithm
def evaluate_factor_model(U, V, test_df):
    preds = np.sum(
        U[test_df["user_idx"].values] * V[test_df["movie_idx"].values],
        axis=1
    )

    preds = np.clip(preds, 0.5, 5.0)
    y_true = test_df["rating"].values

    rmse = rmse_from_predictions(y_true, preds)
    mae = mae_from_predictions(y_true, preds)

    return rmse, mae