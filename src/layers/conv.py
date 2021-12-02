import numpy as np
from typing import Tuple

from layers.layer import Layer


class Conv2D(Layer):
    """
    Convolutional layer

    Args:
        kernel_size (int): Size of the convolutional kernel
        filters (int): Number of filters
        stride (int): Stride of the convolutional kernel
        padding (str): Padding type
        activation (str): Activation function
        use_bias (bool): Whether to use bias
        name (str): Name of the layer
    """

    __kernel_size: Tuple[int, int]
    __stride: int
    __padding: int
    __weights: np.ndarray
    __bias: np.ndarray
    __output_shape: Tuple[int, int]
    __input_shape: Tuple[int, int]

    def __init__(self,
                 kernel_size: tuple,
                 filters: int,
                 stride: int = 1,
                 padding: bool = False,
                 activation: str = None,
                 use_bias: bool = True,
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
        self.__use_bias = use_bias

    # Getters
    # -------------------------------------------------------------------------

    @property
    def kernel_size(self) -> tuple:
        return self.__kernel_size

    @property
    def filters(self) -> int:
        return self.__filters

    @property
    def params(self) -> list:
        return [self.__weights, self.__bias]

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

    @property
    def bias(self) -> np.ndarray:
        return self.__bias

    @property
    def activation(self) -> str:
        return self.__activation

    @property
    def output(self) -> np.ndarray:
        return self.__output

    @property
    def input(self) -> np.ndarray:
        return self.__input

    @property
    def use_bias(self):
        return self.__use_bias

    # Setters
    # -------------------------------------------------------------------------

    @weights.setter
    def weights(self, weights: np.ndarray):
        self.__weights = weights

    @bias.setter
    def bias(self, bias: np.ndarray):
        self.__bias = bias

    @params.setter
    def params(self, params: list):
        self.__weights = params[0]
        self.__bias = params[1]

    @input.setter
    def input(self, x: np.ndarray):
        self.__input = x

    @use_bias.setter
    def use_bias(self, use_bias: bool):
        self.__use_bias = use_bias

    # Methods
    # -------------------------------------------------------------------------

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

    def backward(self, gradients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass
        Args:
            gradients: gradients of the loss with respect to the output of this layer
        Returns:
            gradients with respect to the input of this layer
        """
        x = self.__input
        gradients = gradients.reshape((x.shape[0], x.shape[1], x.shape[2], self.__filters))
        gradients_w = np.zeros(self.__weights.shape)
        gradients_b = np.zeros(self.__bias.shape)
        for i in range(x.shape[0]):  # batch
            for j in range(x.shape[1]):  # height
                for k in range(x.shape[2]):  # width
                    for L in range(self.__filters):  # filters
                        gradients_w[L] += np.sum(
                            x[i, j:j + self.__kernel_size[0], k:k + self.__kernel_size[1]] * gradients[i, j, k, L])
                        gradients_b[L] += gradients[i, j, k, L]

        return gradients_w, gradients_b

    @staticmethod
    def convolve_2d(image: np.ndarray, kernel: np.ndarray, padding: bool = False, stride: int = 1) -> np.ndarray:
        """
        Performs a valid convolution on an image with a kernel.

        Args:
            image: A grayscale image.
            kernel: A kernel.
            padding: Whether to pad the image.
            stride: convolution stride size.

        Returns:
            A grayscale image.
        """
        # Get the dimensions of the image and kernel
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # Pad the image if padding is True
        if padding:
            image = np.pad(image, ((kernel_height // 2, kernel_height // 2), (kernel_width // 2, kernel_width // 2)),
                           mode='constant', constant_values=0)

        # Create the output image
        output_height = (image_height - kernel_height) // stride + 1
        output_width = (image_width - kernel_width) // stride + 1
        convolved_image = np.zeros((output_height, output_width))

        # Perform the convolution
        for i in range(output_height):
            for j in range(output_width):
                convolved_image[i, j] = np.sum(
                    image[i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width] * kernel)

        return convolved_image

    def summary(self):
        print(f"Layer: {self.name}, Output shape: {self.output.shape}")
