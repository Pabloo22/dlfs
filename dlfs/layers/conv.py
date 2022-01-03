import numpy as np
from typing import Tuple

from .layer import Layer
from dlfs.activation_functions import get_activation_function, ActivationFunction


class Conv2D(Layer):
    """
    Convolutional layer

    Args:
        kernel_size (int): Size of the convolutional kernel
        n_filters (int): Number of filters
        stride (int): Stride of the convolutional kernel
        padding (str): Padding type
        activation (ActivationFunction): Activation function
        use_bias (bool): Whether to use bias
        name (str): Name of the layer
    """

    __kernel_size: Tuple[int, int]
    __n_filters: int
    __stride: int
    __padding: int
    __weights: np.ndarray
    __bias: np.ndarray
    __activation: ActivationFunction
    __use_bias: bool

    def __init__(self,
                 kernel_size: tuple,
                 n_filters: int,
                 stride: int = 1,
                 padding: bool = False,
                 activation: str = None,
                 use_bias: bool = True,
                 name: str = None):

        super().__init__(name=name)
        self.__kernel_size = kernel_size
        self.__n_filters = n_filters
        self.__stride = stride
        self.__padding = padding
        self.__weights = np.random.randn(n_filters, *kernel_size)
        self.__bias = np.random.randn(n_filters)
        self.__input = None
        self.__output = None
        self.__activation = get_activation_function(activation)
        self.__use_bias = use_bias

    # Getters
    # -------------------------------------------------------------------------

    @property
    def kernel_size(self) -> tuple:
        return self.__kernel_size

    @property
    def filters(self) -> int:
        return self.__n_filters

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
    def activation(self) -> ActivationFunction:
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

    @property
    def input_shape(self) -> tuple:
        return self.input_shape

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

    @input_shape.setter
    def input_shape(self, input_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = self._get_output_shape()

    @use_bias.setter
    def use_bias(self, use_bias: bool):
        self.__use_bias = use_bias

    # Methods
    # -------------------------------------------------------------------------

    def initialize(self, input_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = self._get_output_shape()
        self.initialized = True

    @staticmethod
    def simple_convolution(image: np.ndarray, kernel: np.ndarray, padding: bool = False, stride: int = 1) -> np.ndarray:
        """
        Performs a valid convolution on an image (with only a channel) with a kernel.

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
                    image[i * stride:i * stride + kernel_height,
                    j * stride:j * stride + kernel_width] * kernel)

        return convolved_image

    @staticmethod
    def convolve(image: np.ndarray,
                 kernel: np.ndarray,
                 bias: np.ndarray = None,
                 padding: bool = False,
                 stride: int = 1) -> np.ndarray:
        """
        Performs a valid convolution to an image with a kernel.

        Args:
            image: An image with multiple channels.
            kernel: A kernel tensor of shape (n_channels, kernel_height, kernel_width).
            bias: A bias tensor of shape (n_channels,).
            padding: Whether to pad the image.
            stride: convolution stride size.

        Returns:
            A tensor of shape (n_channels, output_height, output_width).
        """

        # Get the dimensions of the image and kernel
        n_channels, image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape[1:]

        # Pad the image if padding is True
        if padding:
            image = np.pad(image, ((0, 0), (kernel_height // 2, kernel_height // 2),
                                   (kernel_width // 2, kernel_width // 2)),
                           mode='constant', constant_values=0)

        # Create the output image
        output_height = (image_height - kernel_height) // stride + 1
        output_width = (image_width - kernel_width) // stride + 1

        # Initialize the output tensor
        convolved_image = np.zeros((n_channels, output_height, output_width))

        # Perform the convolution



    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass
        Args:
            x: input data
            training: for compatibility with other layers
        Returns:
            output data
        """

        output = np.zeros((x.shape[0], x.shape[1], x.shape[2], self.__n_filters))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for L in range(self.__n_filters):
                        output[i, j, k, L] = np.sum(
                            x[i, j:j + self.__kernel_size[0], k:k + self.__kernel_size[1]] * self.__weights[L]) + \
                                             self.__bias[L]

        self.__input = x
        self.__output = output
        return output

    def get_delta(self, last_delta: np.ndarray, dz_da: np.ndarray) -> np.ndarray:
        """
        Backward pass
        Args:
            last_delta: gradients of the loss with respect to the output of this layer
            dz_da: derivative of the z of the next layer (i+1) with respect to the activation of the current layer (i)
        Returns:
            The corresponding delta of the layer (d_cost/d_z).
        """


    def count_params(self) -> int:
        """
        Counts the number of parameters of this layer
        Returns:
            number of parameters
        """
        return self.__weights.size + self.__bias.size

    def _get_output_shape(self) -> tuple:
        """
        Get the output shape
        Returns:
            tuple: output shape
        """
        return (self.input_shape[0], self.input_shape[1] - self.__kernel_size[0] + 1,
                self.input_shape[2] - self.__kernel_size[1] + 1, self.__n_filters)

    def compute_weights_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """

        Args:
             gradients: The gradients of the loss with respect to the output of this layer.
        Returns:
            The gradients of the loss with respect to the weights of this layer.
        """
        return self.activation.gradient(gradients)

    def summary(self):
        print(f"Layer: {self.name}, Output shape: {self.output.shape}")
