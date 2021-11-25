import numpy as np

from base import Layer


class MaxPooling2D(Layer):

    def __init__(self, input_shape, pool_size=2, stride=2, padding=0, name="MaxPooling2D"):

        output_shape = ...
        super(MaxPooling2D, self).__init__(input_shape, output_shape, name)

    def forward(self, x):
        pass

    def backward(self, dout):
        pass
