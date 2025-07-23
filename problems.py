from sklearn import datasets
import numpy as np


def make_moons_with_ratio(n_samples=100, ratio_for_positive=0.7, noise_level=0.1, horizontally_stretch = 1):
    X, y = datasets.make_moons(n_samples=int(n_samples*2*ratio_for_positive), noise=noise_level, random_state=20)
    X_positive = X[y==1]
    y_positive = y[y==1]
    X, y = datasets.make_moons(n_samples=int(n_samples*2*(1-ratio_for_positive)), noise=noise_level, random_state=400)
    X_negative = X[y==0]
    y_negative = y[y==0]
    X = np.concatenate((X_positive, X_negative), axis=0)
    y = np.concatenate((y_positive, y_negative), axis=0)
    X[:, 0] = X[:, 0] * horizontally_stretch
    return X, y


def filter_dataset_np(X, y,
                      pos_x_interval=(-np.inf, np.inf), pos_y_interval=(-np.inf, np.inf),
                      neg_x_interval=(-np.inf, np.inf), neg_y_interval=(-np.inf, np.inf)):
    """
    Filters the dataset based on specified intervals for positive and negative points.

    Args:
        X (np.ndarray): Array of shape (n_samples, 2)
        y (np.ndarray): Array of shape (n_samples,)
        pos_x_interval (tuple): Interval for x of positive points
        pos_y_interval (tuple): Interval for y of positive points
        neg_x_interval (tuple): Interval for x of negative points
        neg_y_interval (tuple): Interval for y of negative points

    Returns:
        filtered_X (np.ndarray): Filtered points
        filtered_y (np.ndarray): Corresponding labels
    """
    # Create boolean masks for positive points
    pos_mask = (y == 1)
    pos_x_mask = (X[:, 0] >= pos_x_interval[0]) & (X[:, 0] <= pos_x_interval[1])
    pos_y_mask = (X[:, 1] >= pos_y_interval[0]) & (X[:, 1] <= pos_y_interval[1])

    # Create boolean masks for negative points
    neg_mask = (y == 0)
    neg_x_mask = (X[:, 0] >= neg_x_interval[0]) & (X[:, 0] <= neg_x_interval[1])
    neg_y_mask = (X[:, 1] >= neg_y_interval[0]) & (X[:, 1] <= neg_y_interval[1])

    # Combine masks
    pos_indices = pos_mask & pos_x_mask & pos_y_mask
    neg_indices = neg_mask & neg_x_mask & neg_y_mask

    # Concatenate indices
    combined_mask = pos_indices | neg_indices

    return X[combined_mask], y[combined_mask]


def create_domain_adaptation_problem(n_samples=300, noise_level=0.1,
                                     theta = np.pi / 4,
                                    horizontally_stretch = 1,
                                    source_ratio=0.5, target_ratio=0.5,
                                    pos_x_interval_source=(-np.inf, np.inf), pos_y_interval_source=(-np.inf, np.inf),
                                    neg_x_interval_source=(-np.inf, np.inf), neg_y_interval_source=(-np.inf, np.inf),
                                    pos_x_interval_target = (-np.inf, np.inf), pos_y_interval_target = (-np.inf, np.inf),
                                    neg_x_interval_target = (-np.inf, np.inf), neg_y_interval_target = (-np.inf, np.inf),
                                     ):
    X_s, y_s = make_moons_with_ratio(n_samples, source_ratio, noise_level, horizontally_stretch)
    X_t, y_t = make_moons_with_ratio(n_samples, target_ratio, noise_level, horizontally_stretch)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    X_t = np.dot(X_t, rotation_matrix)
    X_s, y_s = filter_dataset_np(X_s, y_s, pos_x_interval_source, pos_y_interval_source, neg_x_interval_source, neg_y_interval_source)
    X_t, y_t = filter_dataset_np(X_t, y_t, pos_x_interval_target, pos_y_interval_target, neg_x_interval_target, neg_y_interval_target)
    return X_s, y_s, X_t, y_t