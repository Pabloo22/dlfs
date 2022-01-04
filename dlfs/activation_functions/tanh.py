import numpy as np

from .activation_function import ActivationFunction


class Tanh(ActivationFunction):

    def __init__(self):
        super().__init__(name='tanh')

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        return 1 - np.square(z)
