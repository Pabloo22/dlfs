import numpy as np

from .activation_function import ActivationFunction


class Sigmoid(ActivationFunction):

    def __init__(self):
        super().__init__(
            name='sigmoid',
            description='Sigmoid activation function',
            function=lambda x: 1 / (1 + np.exp(-x)),
            derivative=lambda x: self.function(x) * (1 - self.function(x)))
