import numpy as np
from typing import Tuple

from dlfs.optimizers import Optimizer


class SGD(Optimizer):
    """
    Mini-batch stochastic gradient descent.
    """

    def __init__(self, learning_rate: float = 0.01):
        super(SGD, self).__init__(learning_rate=learning_rate)

    def update(self, parameters: Tuple[np.ndarray, np.ndarray],
               gradients: Tuple[np.ndarray, np.ndarray]):
        """
        Update parameters using SGD

        Args:
            parameters (np.ndarray): parameters to be updated (weights and bias)
            gradients (np.ndarray): gradients of parameters (weights and bias)
        """
        w, b = parameters
        dw, db = gradients

        w -= self.learning_rate * dw
        b -= self.learning_rate * db
