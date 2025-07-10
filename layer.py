import numpy as np

from activations import sigmoid, d_sig


class Layer:
    def __init__(self, n_neurons, activation=sigmoid):
        self._n_neurons = n_neurons

        self._activation = activation

        # self._w = self._init_weights(n_neurons, prev_layer_size)
        # self._b = self._init_bias()
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
        dz = da * d_sig(self._a)
        dw = np.dot(dz, self._a_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        da_ret = np.dot(self._w.T, dz)

        self.update_params(dw, db, alpha)

        return da_ret

    def update_params(self, dw, db, alpha):
        self._w -= alpha * dw
        self._b -= alpha * db
