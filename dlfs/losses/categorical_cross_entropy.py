import numpy as np

from .loss_function import LossFunction


class CategoricalCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def __init__(self, name="cross_entropy"):
        super(CategoricalCrossEntropy, self).__init__(name)

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """

        Args:
            y_pred: (np.array)
            y_true: (np.array)

        Returns:
            (float)
        """
        return -np.sum(y_true * np.log(y_pred))

    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """

        Args:
            y_pred: (np.array)
            y_true: (np.array)

        Returns:
            (np.array)
        """
        return -y_true / y_pred
