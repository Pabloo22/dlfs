from abc import ABC, abstractmethod
import numpy as np


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
    def update(self, parameters: np.ndarray, gradients: np.ndarray):
        """Update parameters based on gradients.
        Args:
            parameters: Parameters to update.
            gradients: Gradients of the parameters.
        """
        raise NotImplementedError