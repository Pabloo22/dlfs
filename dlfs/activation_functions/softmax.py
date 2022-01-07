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
        return np.mean(exp_x / np.sum(exp_x))

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the softmax of the input.
        Args:
            z: The input to the softmax function. It has shape (batch_size, num_classes).
        """
        batch_size = z.shape[0]
        jacobian = np.empty((batch_size, z.shape[1], z.shape[1]))
        softmax = Softmax.forward(z)

        # We can use np.einsum to compute the jacobian in a more efficient way. Than just using:
        # for m in range(batch_size):
        #     for i in range(z.shape[1]):
        #         for j in range(z.shape[1]):
        #             jacobian[m, i, j] = softmax[m, i] * ((i == j) - softmax[m, j])
        for m in range(batch_size):
            for i in range(z.shape[1]):
                for j in range(z.shape[1]):
                    jacobian[m, i, j] = softmax[m, i] * ((i == j) - softmax[m, j])

        return jacobian
