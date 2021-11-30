import numpy as np


class MSE:
    """
    Class that defines the Mean Squared Error loss function
    """

    @staticmethod
    def loss(y_true, y_pred):
        """
        Calculates the Mean Squared Error loss

        """
        return np.sum(np.square(y_true - y_pred)) / y_true.shape[0]

    @staticmethod
    def gradient(y_true, y_pred):
        """
        Calculates the gradient of the Mean Squared Error loss
        """
        return y_pred - y_true
