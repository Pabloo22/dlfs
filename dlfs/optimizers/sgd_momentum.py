import numpy as np

from .optimizer import Optimizer


class SGDMomentum(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        super(SGDMomentum, self).__init__(lr)
        self.momentum = momentum
        self.__v = None

    def update(self, params, grads):
        w, b = params
        dw, db = grads

        if self.__v is None:
            self.__v = (np.zeros_like(w), np.zeros_like(b))

        vw, vb = self.__v

        # update velocity (weighted average)
        vw = self.momentum * vw + (1 - self.momentum) * dw
        vb = self.momentum * vb + (1 - self.momentum) * db

        # update parameters
        w -= self.learning_rate * vw
        b -= self.learning_rate * vb

        self.__v = (vw, vb)
