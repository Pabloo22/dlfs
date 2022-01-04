from . import ActivationFunction


class ReLU(ActivationFunction):

    def __init__(self):
        super().__init__(name='ReLU',
                         function=lambda x: x * (x > 0),
                         derivative=lambda x: 1. * (x > 0))
