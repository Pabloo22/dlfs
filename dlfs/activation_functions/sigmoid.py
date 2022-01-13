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
"""Contains the sigmoid activation function"""

import numpy as np
from .activation_function import ActivationFunction


class Sigmoid(ActivationFunction):
    """Sigmoid activation function.

    The sigmoid activation function is defined as:
    f(x) = 1 / (1 + e^(-x))

    It returns a value between 0 and 1.
    """

    def __init__(self):
        super().__init__(name='sigmoid')

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """Computes the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the sigmoid activation function."""
        sigmoid = Sigmoid.forward(z)
        return sigmoid * (1 - sigmoid)
