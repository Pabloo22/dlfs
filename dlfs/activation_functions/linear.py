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
"""Contains the Linear Activation function"""

import numpy as np
from . import ActivationFunction


class Linear(ActivationFunction):
    """Linear activation function.

    This activation function is the identity function, it does not change the
    input.
    """
    def __init__(self):
        super().__init__(name='linear')

    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """Returns a tensor of the same shape as z, with all elements set to 1."""
        return np.ones_like(z)
