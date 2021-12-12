from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """
    Base class for activation functions.
    """

    def __init__(self, name, description, function, derivative):
        self.name = name
        self.description = description
        self.__function = function
        self.__derivative = derivative

    # Getters
    # -------------------------------------------------------------------------

    @property
    def function(self):
        return self.__function

    @property
    def derivative(self):
        return self.__derivative

    # Methods
    # -------------------------------------------------------------------------

    def forward(self, x):
        raise self.forward(x)

    def gradient(self, x):
        raise self.gradient(x)

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return self.name
