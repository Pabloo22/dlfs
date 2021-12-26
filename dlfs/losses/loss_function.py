from abc import ABC, abstractmethod
from numpy import ndarray


class LossFunction(ABC):
    """
    Base class for loss functions.
    """

    def __init__(self, name):
        self.name = name

    @staticmethod
    @abstractmethod
    def compute_loss(y_true, y_pred) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gradient(y_true, y_pred) -> ndarray:
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        return self.compute_loss(y_true, y_pred)
