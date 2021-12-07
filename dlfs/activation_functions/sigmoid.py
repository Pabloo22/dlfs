import numpy as np

from .activation_function import ActivationFunction


class Sigmoid(ActivationFunction):

    def __init__(self):
        super().__init__()
        self.name = 'sigmoid'
        self.description = 'Sigmoid activation function'
        self.__function = lambda x: 1 / (1 + np.exp(-x))
        self.__derivative = lambda x: self.__function(x) * (1 - self.__function(x))

    # Getters
    # --------------------------------------------------------------------------------------------------------------

    @property
    def function(self):
        return self.__function

    @property
    def derivative(self):
        return self.__derivative

    # Methods
    # --------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        return self.__function(x)

    def gradient(self, x):
        return self.__derivative(x)

    def __str__(self):
        return self.name
