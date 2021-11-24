import numpy as np
from typing import Tuple

from base import Layer


class Conv2D(Layer):
    """
    Convolutional layer
    """

    __kernel_size: Tuple[int, int]
    __stride: int
    __padding: int
    __weights: np.ndarray
    __bias: np.ndarray
    __output_shape: Tuple[int, int]
    __input_shape: Tuple[int, int]


    def __init__(self, kernel_size: tuple, filters: int, stride: int = 1, padding: int = 0, activation: str = None,
                 name: str = None):

        super().__init__(input_shape=(2, 2), output_shape=(1, 1), name=name)
        self.__kernel_size = kernel_size
        self.__filters = filters
        self.__stride = stride
        self.__padding = padding
        self.__weights = np.random.randn(filters, *kernel_size)
        self.__bias = np.random.randn(filters)
        self.__input = None
        self.__output = None

        self.__activation = activation

    @property
    def kernel_size(self) -> tuple:
        return self.__kernel_size

    @property
    def filters(self) -> int:
        return self.__filters

    @property
    def params(self) -> list:
        return [self.__weights, self.__bias]

    @params.setter
    def params(self, params: list):
        self.__weights = params[0]
        self.__bias = params[1]

    @property
    def n_params(self):
        return self.__weights.size + self.__bias.size

    @property
    def shape(self) -> tuple:
        return self.__output.shape

    @property
    def stride(self) -> int:
        return self.__stride

    @property
    def padding(self) -> int:
        return self.__padding

    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    @weights.setter
    def weights(self, weights: np.ndarray):
        self.__weights = weights

    @property
    def bias(self) -> np.ndarray:
        return self.__bias

    @bias.setter
    def bias(self, bias: np.ndarray):
        self.__bias = bias

    @property
    def activation(self) -> str:
        return self.__activation

    @property
    def output(self) -> np.ndarray:
        return self.__output

    @property
    def input(self) -> np.ndarray:
        return self.__input

    @input.setter
    def input(self, x: np.ndarray):
        self.__input = x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass
        Args:
            x: input data
        Returns:
            output data
        """

        output = np.zeros((x.shape[0], x.shape[1], x.shape[2], self.__filters))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for L in range(self.__filters):
                        output[i, j, k, L] = np.sum(
                            x[i, j:j + self.__kernel_size[0], k:k + self.__kernel_size[1]] * self.__weights[L]) + \
                                             self.__bias[L]
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

        gradients = gradients.reshape((x.shape[0], x.shape[1], x.shape[2], self.__filters))
        gradients_w = np.zeros(self.__weights.shape)
        gradients_b = np.zeros(self.__bias.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for L in range(self.__filters):
                        gradients_w[L] += x[i, j:j + self.__kernel_size[0], k:k + self.__kernel_size[1]] * gradients[
                            i, j, k, L]
                        gradients_b[L] += gradients[i, j, k, L]
        return gradients_w, gradients_b

    def summary(self):
        print(f"Layer: {self.name}, Output shape: {self.output.shape}")