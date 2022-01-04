import numpy as np

from .activation_function import ActivationFunction


class Sigmoid(ActivationFunction):

    def __init__(self):
        super().__init__(name='sigmoid')

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        sigmoid = Sigmoid.forward(z)
        return sigmoid * (1 - sigmoid)
