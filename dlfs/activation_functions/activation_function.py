import numpy as np

from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """
    Base class for activation functions.
    """

    def __init__(self, name):
        self.name = name

    # Methods
    # -------------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the activation function.

        Args:
            x: Input of the activation function.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the activation function.

        Args:
            z: Input of the activation function.
        """
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return self.name
