import numpy as np

from .metric import Metric


class Accuracy(Metric):

    def __init__(self, name='accuracy'):
        super().__init__(name)

    @staticmethod
    def compute_metric(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the accuracy of the model.

        Args:
            y_true: The true labels.
            y_pred: The predicted labels.

        Returns:
            The accuracy of the model.
        """
        return np.mean(y_true == y_pred)
