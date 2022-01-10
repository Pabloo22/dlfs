import numpy as np
from typing import Union

from .layer import Layer


class MaxPool2D(Layer):
    """
    MaxPool2D class
    """

    def __init__(self,
                 pool_size: Union[int, tuple] = 2,
                 stride: Union[int, tuple] = 2,
                 padding: Union[int, tuple] = 0,
                 name="MaxPool2D",
                 activation: str = None,
                 input_shape: tuple = None,):
        """
        MaxPool2D class constructor.
        Args:
            pool_size: size of the pooling window
            stride: stride of the pooling window
            padding: if True, add padding to the input
            name: name of the layer
            activation: activation function of the layer
            input_shape: shape of the input tensor
        """
        input_shape = (None, *input_shape) if input_shape is not None else None
        pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        # compute output shape (batch_size, width, height, n_channels)

        output_shape = self._get_output_shape(input_shape) if input_shape is not None else None
        super().__init__(name=name, activation=activation, input_shape=input_shape, output_shape=output_shape)

    def _get_output_shape(self, input_shape: tuple) -> tuple:
        """
        Given the input shape, return the output shape.
        Args:
            input_shape: shape of the input tensor (n_filters, width, height)
        """

        return (input_shape[0],  # batch_size
                int((input_shape[1] - self.pool_size[0]) / self.stride[0]) + 1,  # width
                int((input_shape[2] - self.pool_size[1]) / self.stride[1]) + 1,  # height
                input_shape[3])  # n_channels

    def initialize(self, input_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = self._get_output_shape(input_shape)
        self.initialized = True

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass of the layer.
        Args:
            x: input numpy array
            training: training mode (added for compatibility with the base Layer class)
        """
        self.inputs = x
        self.z = np.zeros(self.output_shape)
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                self.z[:, i, j, :] = np.max(x[:, i * self.stride[0]:i * self.stride[0] + self.pool_size[0],
                                              j * self.stride[1]:j * self.stride[1] + self.pool_size[1], :],
                                            axis=(1, 2))
        return self.z if self.activation is None else self.activation(self.z)

    def get_delta(self, last_delta: np.ndarray, dz_da: np.ndarray) -> np.ndarray:
        # TODO: Implement backward pass
        pass

    def summary(self) -> str:
        return f"{self.name} ({self.input_shape} -> {self.output_shape})"
