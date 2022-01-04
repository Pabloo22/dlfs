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
        """

        # create initial gradient matrix
        gradients = np.empty_like(z)

        # compute the softmax of the input
        softmax = Softmax.forward(z)

        # compute the gradient of the softmax
        for i in range(len(z)):
            for j in range(len(z[i])):
                if j == np.argmax(z[i]):
                    gradients[i][j] = softmax[i] * (1 - softmax[i])
                else:
                    gradients[i][j] = -softmax[i] * softmax[j]

        return gradients
