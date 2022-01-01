import numpy as np

from .layer import Layer


class Flatten(Layer):
    """
    Flatten layer. It flattens the input to a 1D vector.
    """

    def __init__(self, name: str = "Flatten"):

        super(Flatten, self).__init__(name=name)

    def initialize(self, input_shape: tuple):
        """
        Initialize the layer.
        Args:
            input_shape: The input shape.
        """
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.initialized = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass of the layer.
        Args:
            x: Input to the layer.
            training: For compatibility with the base class.
        Returns:
            A 1D vector.
        """
        return np.reshape(x, self.output_shape)

    def get_delta(self, last_delta: np.ndarray) -> np.ndarray:
        return np.reshape(last_delta, self.input_shape)

    def summary(self) -> str:
        return f"{self.name} ({self.input_shape} -> {self.output_shape})"
