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
"""Contains the Rectified Linear Unit (ReLU) activation function"""

import numpy as np
from . import ActivationFunction


class ReLU(ActivationFunction):
    """Rectified Linear Unit (ReLU) activation function.

    The ReLU activation function is the most common activation function in deep
    learning. It is a linear function that is bounded above by 0. One of the positive
    features of the ReLU activation function is that it is fast to compute and that helps
    avoid vanishing gradients.
    """

    def __init__(self):
        super().__init__(name='relu')

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """Computes the forward pass of the ReLU activation function."""

        # (x > 0) returns a boolean array, where True represents the elements of x
        # that are greater than 0. True is cast to 1.0 and False is cast to 0.0 when
        # the array is multiplied by x.
        return x * (x > 0)

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the ReLU activation function."""

        # z is the input to the activation function.
        # (z > 0) returns a boolean array, where True represents the elements of z
        # that are greater than 0. True is cast to 1.0 and False is cast to 0.0 when
        # the array is multiplied by z.
        return 1. * (z > 0)
