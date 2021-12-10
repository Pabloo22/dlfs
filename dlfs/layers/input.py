import numpy as np

from .layer import Layer


class Input(Layer):

    def __init__(self, input_shape: tuple, name="input"):
        super().__init__(input_shape, input_shape, name)
        self.input = None
        self.output = None

    def initialize(self, input_shape: tuple):
        """
        Implemented for consistency with other layers.
        """
        self.initialized = True

    def forward(self, x: np.ndarray):
        self.input = x
        self.output = x
        return x

    def backward(self, gradients: np.ndarray):
        return gradients

    def summary(self) -> str:
        return f"Input: {self.input_shape}"
