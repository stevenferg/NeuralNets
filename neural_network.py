from typing import Sequence
import numpy as np

from losses import bin_cross_entropy_cost
from layer import Layer


class NeuralNetwork:
    """
    Neural Network

    NeuralNetwork trains a neural net with parameters (w_1, b_1, ... , w_n, b_n)
    with n corresponding to the number of layers passed to __init__.

    Parameters
    ----------
    layers: Iterable of type Layer
        The layers of the neural network to be trained.

    Attributes
    ----------
    _layers: Iterable of type Layer
        The layers of the neural network
    """

    def __init__(self, layers: Sequence[Layer]):
        self._layers = layers

    def train(self, X, y, iters=10000, learning_rate=0.01):
        features_out = X.shape[0]
        for layer in self._layers:
            layer.init_params(features_out)
            features_out = layer._n_neurons

        for i in range(iters):
            a = X

            for layer in self._layers:
                a = layer.forward(a)

            cost = bin_cross_entropy_cost(y, a)

            if i % 1000 == 0:
                print(f"Iter {i:0>4} -\t Cost {cost}")

            # dz = a - y
            da = -(y / a) + ((1 - y) / (1 - a))

            for layer in reversed(self._layers):
                da = layer.backward(da, learning_rate, y.shape[1])
                # dz = da * d_sig(layer._a_prev)

    def predict(self, X):
        a = X
        for layer in self._layers:
            a = layer.forward(a)

        return np.where(a > 0.5, 1, 0)
