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
"""Contains the stochastic gradient descent momentum optimizer."""

import numpy as np
from typing import Tuple
from .optimizer import Optimizer
from dlfs.layers import Layer


class SGDMomentum(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super(SGDMomentum, self).__init__(learning_rate)
        self.momentum = momentum
        self.__v = {}

    def update(self, layer: Layer, gradients: Tuple[np.ndarray, np.ndarray]):
        """Updates the weights of the layer using the delta.

        Args:
            layer: The layer to update.
            gradients: The delta of the layer.
        """
        dw, db = gradients

        vw, vb = self.__v[layer.name]

        # update velocity. It uses exponential weighted average.
        vw = self.momentum * vw + (1 - self.momentum) * dw
        vb = self.momentum * vb + (1 - self.momentum) * db

        # update parameters
        layer.weights -= self.learning_rate * vw
        layer.bias -= self.learning_rate * vb

        # store the velocity
        self.__v[layer.name] = (vw, vb)

    def add_slot(self, layer: Layer):
        """Adds a slot for the layer.

        Args:
            layer: The layer to add a slot for.
        """
        self.__v[layer.name] = (np.zeros_like(layer.weights), np.zeros_like(layer.bias))

    def reset(self):
        """Resets the optimizer."""
        self.__v = {}
