import numpy as np

from layer import Layer


class MaxPooling2D(Layer):

    def __init__(self, input_shape, pool_size=2, stride=2, padding=0, name="MaxPooling2D"):

        output_shape = ...
        super(MaxPooling2D, self).__init__(input_shape, output_shape, name)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: np.ndarray):
        pass

    def backward(self, gradients: np.ndarray):
        pass

    def summary(self) -> str:
        return f"{self.name} ({self.input_shape} -> {self.output_shape})"
