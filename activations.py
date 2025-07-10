import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sig(a):
    return a * (1 - a)


def tanh(z):
    """
    tanh activation function:
        has mean activation of 0 and is a translation and vertical scaling of sigmoid.
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def relu(z):
    return np.maximum(z, 0)


def leaky_relu(z):
    return np.maximum(z, 0.01 * z)
