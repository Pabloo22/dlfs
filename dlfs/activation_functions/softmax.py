import numpy as np

from .activation_function import ActivationFunction


class Softmax(ActivationFunction):
    """
    Softmax activation function.
    """

    def __init__(self):
        super().__init__()
        self.name = "Softmax"
        self.description = "Softmax activation function"
        self.__function = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        self.__derivative = lambda x: self.__function(x) * (1 - self.__function(x))

    # Getters
    # -------------------------------------------------------------------------------------------------
    @property
    def function(self):
        return self.__function

    @property
    def derivative(self):
        return self.__derivative

    # Methods
    # -------------------------------------------------------------------------------------------------

    def forward(self, x):
        return self.__function(x)

    def gradient(self, x):
        return self.__derivative(x)

    def __str__(self):
        return self.name
