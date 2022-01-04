from . import ActivationFunction


class Linear(ActivationFunction):

    def __init__(self):
        super().__init__(
            name='linear',
            function=lambda x: x,
            derivative=lambda x: 1
        )
