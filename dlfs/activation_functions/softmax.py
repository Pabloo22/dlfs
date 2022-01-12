import numpy as np

from .activation_function import ActivationFunction


class Softmax(ActivationFunction):
    """
    Softmax activation function.
    """

    def __init__(self):
        super().__init__(name="softmax")

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """
        Compute the softmax of the input.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the softmax (d_a/d_z) of the input.
        Args:
            z: The input to the softmax function. It has shape (batch_size, num_classes).
        """
        # code adapted from:
        # https://stackoverflow.com/questions/36279904/softmax-derivative-in-numpy-approaches-0-implementation
        softmax = Softmax.forward(z)
        jacobian = - softmax[..., None] * softmax[:, None, :]  # off-diagonal Jacobian
        iy, ix = np.diag_indices_from(jacobian[0])
        jacobian[:, iy, ix] = softmax * (1. - softmax)  # diagonal

        # The code above is equivalent to the following, but is much faster (2x):
        # jacobian = np.empty((outputs.shape[0], outputs.shape[1], outputs.shape[1]))
        # softmax = Softmax.forward(outputs)
        # for m in range(outputs.shape[0]):
        #     for i in range(outputs.shape[1]):
        #         for j in range(outputs.shape[1]):
        #             jacobian[m, i, j] = softmax[m, i] * ((i == j) - softmax[m, j])

        return jacobian
