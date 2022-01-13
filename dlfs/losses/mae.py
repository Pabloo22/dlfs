import numpy as np

from .loss_function import LossFunction


class MAE(LossFunction):
    """
    Class that calculates the Mean Absolute Error loss
    """
    def __init__(self):
        super(MAE, self).__init__(name="mae")

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the Mean Absolute Error loss
        :param y_true: expected output
        :param y_pred: predictions
        :return: Mean Absolute Error loss
        """
        return np.abs(y_true - y_pred).mean()

    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the gradient of the Mean Absolute Error loss with respect to the predictions
        
        Args:
            y_true: labels
            y_pred: predictions

        Returns:
            The gradient of the loss with respect to the predictions.
        """
        return np.sign(y_pred - y_true)
