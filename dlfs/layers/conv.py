import numpy as np
from typing import Tuple, Union

from .layer import Layer
from dlfs.activation_functions import ActivationFunction
from dlfs.convolutions import Convolutioner, get_convolution


class Conv2D(Layer):
    """
    Convolutional layer
    """

    def __init__(self,
                 kernel_size: Union[Tuple[int, int], int],
                 n_filters: int,
                 stride: Union[Tuple[int, int], int] = (1, 1),
                 padding: bool = False,
                 activation: str = None,
                 use_bias: bool = True,
                 mode: str = 'winograd',
                 name: str = "Conv2D",
                 input_shape: tuple = None,
                 weights_init: str = "xavier",
                 bias_init: str = "zeros"):
        """
        Initialize the convolutional layer.
         Args:
            kernel_size (tuple): tuple of 2 integers, specifying the height and width of the 2D convolution window.
            n_filters (int): Number of filters
            stride (tuple or int): specifying the strides of the convolution along the height and width.
                            Can be a single integer to specify the same value for all spatial dimensions
            padding (bool): If True, add padding to the input so that the output has the same shape as the input
                            (assuming stride = 1)
            activation (ActivationFunction): Activation function
            use_bias (bool): Whether to use bias
            name (str): Name of the layer
            input_shape (tuple): shape of the input
            weights_init (str): Initialization method for the weights
            bias_init (str): Initialization method for the bias
        """

        if n_filters <= 0:
            raise ValueError("The number of filters should be greater than 0")
        if len(kernel_size) != 2:
            raise ValueError("The kernel size should be a tuple of two integers")
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError("The kernel size should be greater than 0")
        if stride <= 0:
            raise ValueError("The stride should be greater than 0")
        if mode not in {'winograd', 'simple'}:
            raise ValueError("Unknown convolution mode")

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        input_shape = None if input_shape is None else (None, *input_shape)
        output_shape = None
        padding = (0, 0) if padding is False else (kernel_size[0] // 2, kernel_size[1] // 2)

        # get the output shape if the input shape is known
        if input_shape is not None:
            output_height = (input_shape[1] - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
            output_width = (input_shape[2] - kernel_size[1] + 2 * padding[1]) // stride[1] + 1
            output_shape = (None, output_height, output_width, n_filters)

        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         activation=activation,
                         name=name)

        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.convolution = get_convolution(mode, kernel_size, stride, padding)

    def initialize(self, input_shape: tuple, weights: np.ndarray = None, bias: np.ndarray = None):
        """
        Initialize the layer. Should be called after the input shape is set.

        Args:
            input_shape (tuple): input shape of the layer, it has the form (n_samples, height, width, n_channels)
            weights (np.ndarray): weights of the layer (optional, recommended to be None).
            bias (np.ndarray): bias of the layer (optional, recommended to be None).
        """

        # check if the input shape is correct
        if len(input_shape) != 4:
            raise ValueError("The input shape should be a tuple of four integers: "
                             "(n_samples, height, width, n_channels)")

        self.input_shape = input_shape
        self.output_shape = self._get_output_shape()

        weights_shape = (input_shape[3], self.kernel_size[0], self.kernel_size[1], self.n_filters)
        # initialize weights
        if weights is not None:
            if weights.shape != weights_shape:
                raise ValueError(f"The shape of the weights should be "
                                 "(n_channels_prev_layer, kernel_height, kernel_width, n_channels_current_layer). "
                                 f"Got {weights.shape}, expected {weights_shape}")
            self.weights = weights
        elif self.weights_init == "xavier":
            self.weights = np.random.normal(loc=0,
                                            scale=np.sqrt(1 / (input_shape[3] * self.kernel_size[0] * self.kernel_size[1])),
                                            size=weights_shape)
        elif self.weights_init == "zeros":
            self.weights = np.zeros(weights_shape)
        elif self.weights_init == "ones":
            self.weights = np.ones(weights_shape)
        elif self.weights_init == "uniform":
            self.weights = np.random.uniform(low=-1, high=1, size=weights_shape)
        elif self.weights_init == "normal":
            self.weights = np.random.normal(loc=0, scale=1, size=weights_shape)
        elif self.weights_init == "glorot_uniform":
            self.weights = np.random.uniform(low=-np.sqrt(6 / (input_shape[3] + self.n_filters)),
                                             high=np.sqrt(6 / (input_shape[3] + self.n_filters)),
                                             size=weights_shape)
        else:
            raise ValueError("Unknown weights initialization")

        bias_shape = (self.n_filters,)
        # initialize bias
        if bias is not None:
            if bias.shape != bias_shape:
                raise ValueError(f"The shape of the bias should be "
                                 "(n_channels_current_layer). "
                                 f"Got {bias.shape}, expected {bias_shape}")
            self.bias = bias
        elif self.bias_init == "zeros":
            self.bias = np.zeros(bias_shape)
        elif self.bias_init == "ones":
            self.bias = np.ones(bias_shape)
        elif self.bias_init == "uniform":
            self.bias = np.random.uniform(low=-1, high=1, size=bias_shape)
        elif self.bias_init == "normal":
            self.bias = np.random.normal(loc=0, scale=1, size=bias_shape)
        else:
            raise ValueError("Unknown bias initialization")

        self.initialized = True

    def set_weights(self, weights: np.ndarray = None, bias: np.ndarray = None):
        """
        Set the weights and bias of the layer.

        Args:
            weights (np.ndarray): weights of the layer.
            bias (np.ndarray): bias of the layer.
        """
        if weights is not None:
            # check if the weights shape is correct
            weight_shape = (self.input_shape[3], self.kernel_size[0], self.kernel_size[1], self.n_filters)
            if weights.shape != weight_shape:
                raise ValueError(f"The shape of the weights should be "
                                 "(n_channels_prev_layer, kernel_height, kernel_width, n_channels_current_layer). "
                                 f"Got {weights.shape}, expected {weight_shape}")
            self.weights = weights

        if bias is not None:
            # check if the bias shape is correct
            bias_shape = (self.n_filters,)
            if bias.shape != bias_shape:
                raise ValueError(f"The shape of the bias should be "
                                 "(n_channels_current_layer). "
                                 f"Got {bias.shape}, expected {bias_shape}")

            self.bias = bias

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass
        Args:
            x: input data
            training: for compatibility with other layers
        Returns:
            output data
        """
        self.inputs = x
        self.z = self.convolution.convolve(x, self.weights) + self.bias
        return self.z if self.activation is None else self.activation(self.z)

    def get_delta(self, last_delta: np.ndarray, dz_da: np.ndarray) -> np.ndarray:
        """
        Backward pass
        Args:
            last_delta: gradients of the loss with respect to the output of this layer (dL/dz) if
                this layer is the last layer in the network, or gradients of the loss with respect
                to the output of the next layer (dL/dz_next) if this layer is followed by another
                Conv2D layer.
            dz_da: gradients of the activation with respect to the output of this layer (dL/dz)
        Returns:
            gradients of the loss with respect to the input of this layer (dL/dx)
        """

        # get dL/da


    def count_params(self) -> int:
        """
        Counts the number of parameters of this layer
        Returns:
            number of parameters
        """
        return self.weights.size + self.bias.size

    def _get_output_shape(self) -> tuple:
        """
        Get the output shape
        Returns:
            tuple: output shape
        """
        return (self.input_shape[0],
                self.input_shape[1] - self.kernel_size[0] + 1,
                self.input_shape[2] - self.kernel_size[1] + 1,
                self.n_filters)

    def update(self, optimizer, delta: np.ndarray):
        """
        Update the weights and biases of this layer
        Args:
            optimizer (Optimizer): optimizer used to update the weights and biases
            delta: gradients of the loss with respect to the output of this layer
        """
        # check if the layer is initialized
        if not self.initialized:
            raise ValueError("The layer is not initialized")

        dw = self.convolve(self.inputs, delta, padding=True, stride=self.stride)
        db = np.sum(delta, axis=(1, 2))

        optimizer.update(self, (dw, db))

    def get_dz_da(self) -> np.ndarray:
        pass

    def summary(self):
        print(f"Layer: {self.name}, Output shape: {self.output_shape}")
