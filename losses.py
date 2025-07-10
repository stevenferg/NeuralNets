import numpy as np


def bin_cross_entropy_cost(y, y_hat, eps=1e-6):
    """
    Calculates the Binary Cross Entropy Cost across all training examples
    """
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
