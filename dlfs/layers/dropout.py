import numpy as np

from .layer import Layer
from dlfs.optimizers import Optimizer


class Dropout(Layer):

    def __init__(self, p=0.5, name="Dropout"):
        super(Dropout, self).__init__(name=name, has_weights=False)
        self.p = 1 - p  # probability of keeping a neuron active
        self.mask = None

    def initialize(self, input_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.initialized = True

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, self.p, x.shape) / self.p
            return x * self.mask
        else:
            return x

    def get_delta(self, last_delta, dz_da):
        return last_delta @ dz_da

    def get_dz_da(self):
        return self.mask

    def summary(self):
        return f"{self.name} (p={1 - self.p})"

    def update(self, optimizer: Optimizer, gradients: np.ndarray):
        pass

    def count_params(self) -> int:
        return 0

    def set_weights(self, weights: np.ndarray = None, bias: np.ndarray = None):
        raise NotImplementedError("Dropout layer has no weights")
