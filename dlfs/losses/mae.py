import numpy as np

from .loss_function import LossFunction


class MAE(LossFunction):
    """
    Class that calculates the Mean Absolute Error loss
    """
    def __init__(self):
        super(MAE, self).__init__(name="MAE")

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the Mean Absolute Error loss
        :param y_true: labels
        :param y_pred: predictions
        :return: Mean Absolute Error loss
        """
        return np.mean(np.abs(y_true - y_pred), axis=0)

    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the gradient of the Mean Absolute Error loss
        
        Args:
            y_true: labels
            y_pred: predictions
        
        """
        return np.sign(y_pred - y_true)
