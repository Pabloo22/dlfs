from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):

    def __init__(self, name):
        self.name = name

    @staticmethod
    @abstractmethod
    def compute_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    def __str__(self):
        return self.name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.compute_metric(y_true, y_pred)
