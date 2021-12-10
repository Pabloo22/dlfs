import numpy as np

from .layer import Layer


class Dropout(Layer):

    def __init__(self, p=0.5, name="Dropout"):

        super(Dropout, self).__init__(name=name)
        self.p = p
        self.mask = None

    def initialize(self, input_shape: tuple):
        self.initialized = True

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, self.p, x.shape) / self.p
            return x * self.mask
        else:
            return x

    def backward(self, gradients):
        return gradients * self.mask

    def summary(self):
        return f"{self.name} (p={self.p})"
