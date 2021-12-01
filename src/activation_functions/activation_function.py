from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """
    Base class for activation functions.
    """

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def gradient(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return self.__class__.__name__
