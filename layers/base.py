import numpy as np


class Layer:
    """
    Base class for all layers.
    """

    def __init__(self, input_shape: tuple, output_shape: tuple, activation: str = None, name: str = None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.weights = None
        self.bias = None
        self.name = name

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer.
        Args:
            inputs: input to the layer.
        Returns:
            output of the layer.
        """
        raise NotImplementedError
