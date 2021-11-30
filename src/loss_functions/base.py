

class LossFunction:
    """
    Base class for loss functions.
    """

    def __init__(self, name):
        self.name = name

    @staticmethod
    def loss(y_true, y_pred):
        raise NotImplementedError

    @staticmethod
    def gradient(y_true, y_pred):
        raise NotImplementedError
