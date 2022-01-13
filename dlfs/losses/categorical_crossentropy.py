# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains CategoricalCrossentropy class."""

import numpy as np
from .loss_function import LossFunction


class CategoricalCrossentropy(LossFunction):
    """Cross entropy loss function.

    A loss function used in order to classify 2 or more categories. This class is commonly used to classify more than 2
    categories, such that if is not, binary cross-entropy is used.
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
