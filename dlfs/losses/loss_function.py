from abc import ABC, abstractmethod


class LossFunction(ABC):
    """
    Base class for loss functions.
    """

    def __init__(self, name):
        self.name = name

    @staticmethod
    @abstractmethod
    def compute_loss(y_true, y_pred):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gradient(y_true, y_pred):
        raise NotImplementedError
