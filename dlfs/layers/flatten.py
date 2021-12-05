import numpy as np

from .layer import Layer


class Flatten(Layer):

    def __init__(self, input_shape, name):

        super(Flatten, self).__init__(input_shape=input_shape, output_shape=(np.prod(input_shape),), name=name)

    def forward(self, x):
        pass

    def backward(self, dout):
        pass
