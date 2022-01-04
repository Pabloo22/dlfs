import numpy as np

from .activation_function import ActivationFunction


class Softmax(ActivationFunction):
    """
    Softmax activation function.
    """

    def __init__(self):
        super().__init__(
            name="Softmax",
            function=lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True),
            derivative=lambda x: self.function(x) * (1 - self.function(x)),
        )
