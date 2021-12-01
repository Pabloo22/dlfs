import numpy as np

from layer import Layer


class MaxPooling2D(Layer):

    def __init__(self, input_shape, pool_size=2, stride=2, padding=0, name="MaxPooling2D"):

        output_shape = ...
        super(MaxPooling2D, self).__init__(input_shape, output_shape, name)
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

    def forward(self, x: np.ndarray):
        # TODO: Implement forward pass
        pass

    def backward(self, gradients: np.ndarray):
        # TODO: Implement backward pass
        pass

    def summary(self) -> str:
        return f"{self.name} ({self.input_shape} -> {self.output_shape})"
