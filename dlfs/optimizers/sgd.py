import numpy as np

from dlfs.optimizers import Optimizer


class SGD(Optimizer):
    """
    Mini-batch stochastic gradient descent.
    """

    def __init__(self, lr: float = 0.01):
        super(SGD, self).__init__(learning_rate=lr)

    def update(self, params: np.ndarray, grads: np.ndarray):
        """
        Update parameters using SGD

        Args:
            params (np.ndarray): parameters to be updated (weights and bias)
            grads (np.ndarray): gradients of parameters
        """
        w, b = params
        # The sum is squashing the results down to a single update
        db = np.sum(grads, axis=0, keepdims=True)

        w -= self.learning_rate * grads
        b -= self.learning_rate * db
