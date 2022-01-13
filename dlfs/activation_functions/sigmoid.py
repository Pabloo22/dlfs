import numpy as np

from .activation_function import ActivationFunction


class Sigmoid(ActivationFunction):
    """Sigmoid activation function.

    The sigmoid activation function is defined as:
    f(x) = 1 / (1 + e^(-x))

    It returns a value between 0 and 1.
    """

    def __init__(self):
        super().__init__(name='sigmoid')

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """Computes the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the sigmoid activation function."""
        sigmoid = Sigmoid.forward(z)
        return sigmoid * (1 - sigmoid)
