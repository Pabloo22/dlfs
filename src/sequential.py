import numpy as np
from activation_functions.base import ActivationFunction
from layers.base import Layer


class Sequential:

    def __init__(self, layers=None):
        """
        Initialize a Sequential model
        :param layers: a list of layers
        """
        self.layers = layers
        self.params = {}
        self.grads = {}
        self.cache = {}

    def forward(self, X, start=0):
        """
        :param X:
        :param start:
        :return:
        """
        for i in range(start, len(self.layers)):
            layer = self.layers[i]
            if layer.trainable:
                self.params[layer.name] = layer.params
                self.grads[layer.name] = layer.grads
                self.cache[layer.name] = layer.cache
            X = layer.forward(X)
        return X

    def backward(self, dout):
        """
        :param dout:
        :return:
        """
        for i in reversed(range(0, len(self.layers))):
            layer = self.layers[i]
            dout = layer.backward(dout)
        return dout
