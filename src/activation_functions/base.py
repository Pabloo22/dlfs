

class ActivationFunction:
    """
    Base class for activation functions.
    """

    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return self.__class__.__name__
