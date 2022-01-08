import numpy as np

from .loss_function import LossFunction


class BinaryCrossentropy(LossFunction):

    def __init__(self, name='binary_crossentropy'):
        super().__init__(name)

    @staticmethod
    def compute_loss(y_true, y_pred):
        """
        Compute the binary cross entropy loss.
        """
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return np.mean(-y_true * np.log(y_pred_clipped) - (1 - y_true) * np.log(1 - y_pred_clipped))

    @staticmethod
    def gradient(y_true, y_pred):
        """
        Compute the gradient of the binary cross entropy loss.
        """
        return y_pred - y_true
