import numpy as np

from .loss_function import LossFunction


class CategoricalCrossentropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def __init__(self, name="cross_entropy"):
        super(CategoricalCrossentropy, self).__init__(name)

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Args:
            y_true: the expected distribution of probabilities as a one-hot vector
            y_pred: the predicted distribution of probabilities

        Returns:
            A numpy array with just a single element
        """
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """

        Args:
            y_true: the expected distribution of probabilities as a one-hot vector
            y_pred: the predicted distribution of probabilities

        Returns:
            A numpy array with dimensions (samples, classes)
        """

        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        gradients = -y_true / y_pred_clipped

        return gradients / samples
