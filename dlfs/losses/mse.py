import numpy as np

from .loss_function import LossFunction


class MSE(LossFunction):
    """
    Class that defines the Mean Squared Error loss function
    """

    def __init__(self):
        super(MSE, self).__init__(name="MSE")

    @staticmethod
    def compute_loss(y_true, y_pred):
        """
        Calculates the Mean Squared Error loss
        """
        return np.sum(np.square(y_true - y_pred)) / (2 * y_true.shape[0])

    @staticmethod
    def gradient(y_true, y_pred):
        """
        Calculates the gradient of the Mean Squared Error loss
        """
        return (y_pred - y_true) / y_true.shape[0]
