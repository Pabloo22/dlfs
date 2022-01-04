import numpy as np

from . import ActivationFunction


class Linear(ActivationFunction):

    def __init__(self):
        super().__init__(name='linear')

    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)
