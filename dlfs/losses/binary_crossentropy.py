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
"""Contains BinaryCrossentropy class."""

import numpy as np
from .loss_function import LossFunction


class BinaryCrossentropy(LossFunction):
    """Binary cross-entropy loss function

    A loss used for classify binary categories. This loss function is commonly used with one terminal neuron and a
    sigmoid function.
    """

    def __init__(self, name='binary_crossentropy'):
        super().__init__(name)

    @staticmethod
    def compute_loss(y_true, y_pred):
        """Compute the binary cross entropy loss."""
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return np.mean(-y_true * np.log(y_pred_clipped) - (1 - y_true) * np.log(1 - y_pred_clipped))

    @staticmethod
    def gradient(y_true, y_pred):
        """Compute the gradient of the binary cross entropy loss."""
        return y_pred - y_true
