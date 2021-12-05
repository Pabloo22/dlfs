import numpy as np

from .layer import Layer


class Dropout(Layer):

    def __init__(self, p=0.5, name="Dropout"):

        super(Dropout, self).__init__(None, None, name)
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, self.p, x.shape) / self.p
            return x * self.mask
        else:
            return x

    def backward(self, dout):
        return dout * self.mask
