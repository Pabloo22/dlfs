import numpy as np

from .layer import Layer
from dlfs.optimizers import Optimizer


class Input(Layer):

    def __init__(self, input_shape: tuple, name="input"):
        super().__init__(input_shape, input_shape, name)
        self.input = None
        self.output = None

    def initialize(self, input_shape: tuple, optimizer: Optimizer = None):
        """
        Implemented for consistency with other layers.
        """
        self.initialized = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Implemented for consistency with other layers.

        Args:
            x: Input to the layer.
            training: Whether the model is in training mode.

        Returns:
            Output of the layer (same as input).
        """
        self.input = x
        self.output = x
        return x

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """
        Implemented for consistency with other layers.

        Args:
            gradients: Gradients of the loss function with respect to the output of this layer.

        Returns:
            Gradients of the loss function with respect to the input of this layer.
        """
        return gradients

    def update(self, gradients: np.ndarray):
        """
        Implemented for consistency with other layers.
        """
        pass

    def summary(self) -> str:
        return f"Input: {self.input_shape}"
