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
"""Contains mean square error loss function."""

import numpy as np
from .loss_function import LossFunction


class MSE(LossFunction):
    """Class that defines the Mean Squared Error loss function."""

    def __init__(self):
        super(MSE, self).__init__(name="mse")

    @staticmethod
    def compute_loss(y_true, y_pred):
        """Calculates the Mean Squared Error loss
        
        Args:
            y_true: expected outputs
            y_pred: predictions
        Returns:
            The mean squared error of the batch.

        """
        return np.sum(np.square(y_true - y_pred)) / (2 * y_true.shape[0])

    @staticmethod
    def gradient(y_true, y_pred):
        """Calculates the gradient of the Mean Squared Error loss with respect to the predictions

        Args:
            y_true: labels
            y_pred: predictions

        Returns:
            The gradient of the loss with respect to the predictions.
        """
        return y_pred - y_true
