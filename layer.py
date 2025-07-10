import numpy as np

import activations


class Layer:
    """
    Layer

    A Layer represents a hidden layer or output layer of a neural network.
    It is initialized with a specified number of neurons and an activation function
    to be used in forward and backpropagation.

    Parameters
    ----------
    n_neurons: int
        The number of neurons, or units, in the hidden layer.

    activation: str
        The name of the activation function to be used for the output of the layer.

    Attributes
    ----------
    _n_neurons: int
        The number of neurons in the layer

    _activation: Callable
        The specified activation function

    _d_activation: Callable
        The derivative of the activation function

    _w:
        Matrix of coefficient parameters

    _b:
        Matrix of bias parameters

    _z:
        Computed linear combination (pre-activation)
        wx+b

    _a:
        Computed activation g(z)

    _a_prev:
        Computed activation of previous layer. Needed for backprop.
    """

    def __init__(self, n_neurons, *, activation):
        self._n_neurons = n_neurons

        self._activation, self._d_activation = activations.activation_map[activation]

        self._w = np.zeros((1, 1))
        self._b = np.zeros((n_neurons, 1))

        self._z = None
        self._a = None
        self._a_prev = None

    def init_params(self, n_features):
        self._w = np.random.randn(self._n_neurons, n_features) * 0.01
        self._b = np.zeros((self._n_neurons, 1))

    def forward(self, a_prev):
        self._a_prev = a_prev
        self._z = np.dot(self._w, a_prev) + self._b
        self._a = self._activation(self._z)

        return self._a

    def backward(self, da, alpha, m):
        g_prime = self._d_activation(self._z, self._a)
        dz = da * g_prime
        dw = np.dot(dz, self._a_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        da_ret = np.dot(self._w.T, dz)

        self.update_params(dw, db, alpha)

        return da_ret

    def update_params(self, dw, db, alpha):
        self._w -= alpha * dw
        self._b -= alpha * db
