import numpy as np
from base import Layer
from typing import Tuple


class Conv2D(Layer):
    """
    Convolutional layer
    """

    def __init__(self, kernel_size: tuple, filters, stride=1, padding=0, activation=None, name=None):

        super().__init__(input_shape=(2, 2), output_shape=(1, 1), activation=activation, name=name)
        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(filters, *kernel_size)
        self.bias = np.random.randn(filters)
        self.input = None
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass
        Args:
            x: input data
        Returns:
            output data
        """

        output = np.zeros((x.shape[0], x.shape[1], x.shape[2], self.filters))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for L in range(self.filters):
                        output[i, j, k, L] = np.sum(
                            x[i, j:j + self.kernel_size[0], k:k + self.kernel_size[1]] * self.weights[L]) + self.bias[L]
        return output

    def backward(self, x: np.ndarray, gradients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass
        Args:
            x: input data
            gradients: gradients of the loss with respect to the output of this layer
        Returns:
            gradients with respect to the input of this layer
        """

        gradients = gradients.reshape((x.shape[0], x.shape[1], x.shape[2], self.filters))
        gradients_w = np.zeros(self.weights.shape)
        gradients_b = np.zeros(self.bias.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for L in range(self.filters):
                        gradients_w[L] += x[i, j:j + self.kernel_size[0], k:k + self.kernel_size[1]] * gradients[
                            i, j, k, L]
                        gradients_b[L] += gradients[i, j, k, L]
        return gradients_w, gradients_b
