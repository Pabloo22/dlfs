from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

from dlfs.layers import Layer


class Optimizer(ABC):
    """Base class for all optimizers.

    Args:
        learning_rate (float): learning rate for the optimizer.
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, layer: Layer, gradients: Tuple[np.ndarray, np.ndarray]):
        """Updates parameters based on delta.
        Args:
            layer: The layer to update.
            gradients: Gradients of the parameters passed as a tuple of two numpy arrays composed of the delta of
                the weights and biases.
        """
        raise NotImplementedError

    @abstractmethod
    def add_slot(self, layer: Layer):
        """Adds a slot to the optimizer.

        Args:
            layer: The layer to add a slot to.
        """
        raise NotImplementedError
