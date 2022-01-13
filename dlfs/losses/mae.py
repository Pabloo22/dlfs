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
"""Contains mean absolute error loss function."""

import numpy as np
from .loss_function import LossFunction


class MAE(LossFunction):
    """Class that calculates the Mean Absolute Error loss.
    """
    def __init__(self):
        super(MAE, self).__init__(name="mae")

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates the Mean Absolute Error loss

        Args:
            y_true: expected output
            y_pred: predictions
        Returns:
            The mean absolute error of the batch.
        """
        return np.abs(y_true - y_pred).mean()

    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates the gradient of the Mean Absolute Error loss with respect to the predictions
        
        Args:
            y_true: labels
            y_pred: predictions

        Returns:
            The gradient of the loss with respect to the predictions.
        """
        return np.sign(y_pred - y_true)
