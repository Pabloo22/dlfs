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
"""Home of the base activation function class."""

import numpy as np

from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """Base class for activation functions.

    An activation function takes a tensor as input and returns a tensor of the
    same shape. The activation function is usually applied element-wise.
    """

    def __init__(self, name):
        self.name = name

    # Methods
    # -------------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """Forward pass of the activation function.

        Args:
            x: Input of the activation function.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """Compute the gradient of the activation function.

        Args:
            z: Input of the activation function.
        """
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        return self.name
