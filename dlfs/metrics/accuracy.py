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
        using_multi_label = len(y_true[0]) > 1
        y_pred = np.argmax(y_pred, axis=1) if using_multi_label else np.round(y_pred)
        y_true = np.argmax(y_true, axis=1) if using_multi_label else y_true

        # Return the accuracy of the model (multilabel or binary)
        accuracy = np.mean(y_pred == y_true)
        return accuracy

