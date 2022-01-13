import numpy as np

from .activation_function import ActivationFunction


class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function.

    The hyperbolic tangent activation function is defined as:
    tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    """

    def __init__(self):
        super().__init__(name='tanh')

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """Forward pass of the tanh activation function."""
        # luckily, numpy has a built-in tanh function
        return np.tanh(x)

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the tanh activation function."""
        return 1 - np.square(z)
