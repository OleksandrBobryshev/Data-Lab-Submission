import numpy as np


def build_rating_matrix(df, n_users, n_movies):
    R = np.zeros((n_users, n_movies), dtype=np.float64)

    user_indices = df["user_idx"].to_numpy()
    movie_indices = df["movie_idx"].to_numpy()
    rating_values = df["rating"].to_numpy()

    R[user_indices, movie_indices] = rating_values

    return R

