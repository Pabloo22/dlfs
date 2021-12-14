from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

    @abstractmethod
    def update(self, parameters: Tuple[np.ndarray, np.ndarray], gradients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update parameters based on gradients.
        Args:
            parameters: Parameters to update passed as a tuple of two numpy arrays composed of the weights and biases.
            gradients: Gradients of the parameters passed as a tuple of two numpy arrays composed of the gradients of
                the weights and biases.
        """
        raise NotImplementedError
