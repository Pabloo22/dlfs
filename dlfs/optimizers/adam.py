import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.
    """

    def __init__(self, learning_rate=0.001, decay=0., beta1=0.9, beta2=0.999, epsilon=1e-7):

        super(Adam, self).__init__(learning_rate)
        self.__current_learning_rate = learning_rate
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__epsilon = epsilon
        self.__iterations = 0
        self.__decay = 0
