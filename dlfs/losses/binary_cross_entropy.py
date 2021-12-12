import numpy as np

from .loss_function import LossFunction


class BinaryCrossEntropy(LossFunction):

    def __init__(self, name='binary_cross_entropy'):
        super().__init__(name)

    @staticmethod
    def compute_loss(y_true, y_pred):
        """
        Compute the binary cross entropy loss.
        """
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def gradient(y_true, y_pred):
        """
        Compute the gradient of the binary cross entropy loss.
        """
        return y_pred - y_true
