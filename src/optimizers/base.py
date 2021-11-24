

class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

    def update(self, parameters, gradients):
        """Update parameters based on gradients."""
        raise NotImplementedError
