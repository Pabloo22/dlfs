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
"""Contains the hyperbolic tangent (tanh) activation function"""

import numpy as np
from .activation_function import ActivationFunction


class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function.

    The hyperbolic tangent activation function is defined as:
    tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    """

    def __init__(self):
        super().__init__(name='tanh')

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """Forward pass of the tanh activation function."""
        # luckily, numpy has a built-in tanh function
        return np.tanh(x)

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the tanh activation function."""
        return 1 - np.square(z)
