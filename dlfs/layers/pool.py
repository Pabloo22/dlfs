import numpy as np

from .layer import Layer


class MaxPool2D(Layer):
    """
    MaxPool2D class
    """

    def __init__(self, pool_size=2, stride=2, padding=0, name="MaxPool2D"):

        super(MaxPool2D, self).__init__(name=name)
        self.__pool_size = pool_size
        self.__stride = stride
        self.__padding = padding

    # Getters
    # ------------------------------------------------------------

    @property
    def pool_size(self):
        return self.__pool_size

    @property
    def stride(self):
        return self.__stride

    @property
    def padding(self):
        return self.__padding

    # Methods
    # ------------------------------------------------------------

    def _get_output_shape(self, input_shape: tuple) -> tuple:
        """
        Given the input shape, return the output shape.
        Args:
            input_shape: shape of the input tensor (n_filters, width, height)
        """

        return (input_shape[0],
                input_shape[1] - self.pool_size + 2 * self.padding // self.stride + 1,
                input_shape[2] - self.pool_size + 2 * self.padding // self.stride + 1)

    def initialize(self, input_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = self._get_output_shape(input_shape)
        self.initialized = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer.
        Args:
            x: input numpy array
        """

        output = np.zeros(self.output_shape)

        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                output[:, i, j] = np.max(x[:, i * self.stride: i * self.stride + self.pool_size,
                                              j * self.stride:j * self.stride + self.pool_size], axis=(1, 2))

        return output

    def get_delta(self, last_delta: np.ndarray):
        # TODO: Implement backward pass
        pass

    def summary(self) -> str:
        return f"{self.name} ({self.input_shape} -> {self.output_shape})"
