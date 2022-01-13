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
"""Contains the softmax activation function"""

import numpy as np
from .activation_function import ActivationFunction


class Softmax(ActivationFunction):
    """Softmax activation function.

    The softmax activation function is defined as:
    f(x) = exp(x) / sum(exp(x))

    The softmax activation function is a special case of the exponential
    activation function where the output is constrained to be between 0 and 1. It
    is often used as the last activation function in a network to ensure that
    the output is a probability distribution.

    """

    def __init__(self):
        super().__init__(name="softmax")

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """Compute the softmax of the input."""

        # Softmax is prone to overflow, so we use the following trick, extracted from:
        # https://stackoverflow.com/questions/42599498/numercially-stable-softmax

        # We subtract the max of each row to avoid overflow, thanks to the property that
        # softmax(x) = softmax(x + c) for all c.
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def gradient(z: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the softmax (d_a/d_z) of the input.
        Args:
            z: The input to the softmax function. It has shape (batch_size, num_classes).
        """
        # code adapted from:
        # https://stackoverflow.com/questions/36279904/softmax-derivative-in-numpy-approaches-0-implementation
        softmax = Softmax.forward(z)
        jacobian = - softmax[..., None] * softmax[:, None, :]  # off-diagonal Jacobian
        iy, ix = np.diag_indices_from(jacobian[0])
        jacobian[:, iy, ix] = softmax * (1. - softmax)  # diagonal

        # The code above is equivalent to the following, but is much faster (2x)*:
        # jacobian = np.empty((outputs.shape[0], outputs.shape[1], outputs.shape[1]))
        # softmax = Softmax.forward(outputs)
        # for m in range(outputs.shape[0]):
        #     for i in range(outputs.shape[1]):
        #         for j in range(outputs.shape[1]):
        #             jacobian[m, i, j] = softmax[m, i] * ((i == j) - softmax[m, j])
        #
        # *tested in the mnist

        return jacobian
