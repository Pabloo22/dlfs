import numpy as np

from .optimizer import Optimizer
from dlfs import sequential

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
        self.init_layer()

    def init_layer(self):
        self.__layers = []
        for layer in sequential.layers:
            self.__layers.append((layer,}

    def update(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.bias)
            layer.bias_cache = np.zeros_like(layer.bias)

        layer.weight_momentums = self.__beta1 * layer.weight_momentums + (1 - self.__beta1) * layer.dweights

    # Call once before any parameter updates

