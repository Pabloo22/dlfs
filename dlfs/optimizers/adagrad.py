import numpy as np
from typing import Tuple

from .optimizer import Optimizer


class Adagrad(Optimizer):
    """
    Adagrad optimizer.
    """

    def __init__(self, lr: float = 0.01, eps: float = 1e-8):
        super(Adagrad, self).__init__(lr)
        self.eps = eps  # epsilon to avoid division by zero

    def update(self, parameters: Tuple[np.ndarray, np.ndarray], gradients: Tuple[np.ndarray, np.ndarray]):
        """

        Args:
            parameters: Parameters to update passed as a tuple of two numpy arrays composed of the weights and biases.
            gradients: Gradients of the parameters passed as a tuple of two numpy arrays composed of the delta of
                the weights and biases.

        """
        w, b = parameters
        dw, db = gradients

        




