from typing import Any, Callable
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sig(z, a):
    return a * (1 - a)


def tanh(z):
    """
    tanh activation function:
        has mean activation of 0 and is a translation and vertical scaling of sigmoid.
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def d_tanh(z, a):
    return 1 - (a**2)


def relu(z):
    return np.maximum(z, 0)


def d_relu(z, a):
    return np.where(z < 0, 0, 1)


def leaky_relu(z):
    return np.maximum(z, 0.01 * z)


def d_leaky_relu(z, a):
    return np.where(z < 0, 0.01, 1)


activation_map: dict[str, tuple[Callable, Callable]] = {
    "sigmoid": (sigmoid, d_sig),
    "tanh": (tanh, d_tanh),
    "relu": (relu, d_relu),
    "leaky_relu": (leaky_relu, d_leaky_relu),
}
