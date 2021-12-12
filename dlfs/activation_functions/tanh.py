import numpy as np

from .activation_function import ActivationFunction


class Tanh(ActivationFunction):

    def __init__(self):
        super().__init__(
            name='tanh',
            description='Hyperbolic tangent function',
            function=lambda x: (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x)),
            derivative=lambda x: 1 - np.square(x)
        )
