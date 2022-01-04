import numpy as np

from . import ActivationFunction


class ReLU(ActivationFunction):

    def __init__(self):
        super().__init__(name='relu')

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x * (x > 0)

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        return 1. * (z > 0)
