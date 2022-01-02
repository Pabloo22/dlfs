import numpy as np
from typing import Tuple

from dlfs.optimizers import Optimizer


class SGD(Optimizer):
    """
    Mini-batch stochastic gradient descent.
    """

    def __init__(self, lr: float = 0.01):
        super(SGD, self).__init__(learning_rate=lr)

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

        if dw.shape != w.shape:
            raise ValueError(f"Shape of dw and w do not match: {dw.shape} != {w.shape}")

        w += self.learning_rate * dw
        b += self.learning_rate * db
