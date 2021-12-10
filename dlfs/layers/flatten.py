import numpy as np

from .layer import Layer


class Flatten(Layer):

    def __init__(self, name="Flatten"):

        super(Flatten, self).__init__(name=name)

    def initialize(self, input_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.initialized = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.reshape(x, self.output_shape)

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        return np.reshape(gradients, self.input_shape)

    def summary(self) -> str:
        return f"{self.name} ({self.input_shape} -> {self.output_shape})"
