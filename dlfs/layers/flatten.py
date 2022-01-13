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
"""Home of the Flatten layer."""

import numpy as np
from .layer import Layer


class Flatten(Layer):
    """Flattens the input while preserving the batch dimension.

    This utility layer is responsible for flattening and reducing to one dimension the internal matrix
    of the neural network to later on process a simpler output vector. The main purpose of this layer is to connect
    the last convolutional layer to the fully connected layer. Note that this does not affect the batch size.

    Args:
        name (str): Name of the layer.
    """

    def __init__(self, name: str = "Flatten"):
        super(Flatten, self).__init__(name=name, has_weights=False)

    def initialize(self, input_shape: tuple):
        """
        Initialize the layer. This method is called by the model when the model is being compiled.

        Args:
            input_shape: The input shape.
        """
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.initialized = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass of the layer.

        Args:
            x: Input to the layer.
            training: For compatibility with the base class.

        Returns:
            A 1D vector.
        """
        self.outputs = np.reshape(x, self.output_shape)
        return self.outputs

    def get_d_inputs(self, delta: np.ndarray) -> np.ndarray:
        """Returns the gradient of the loss with respect to the inputs of the layer.

        Args:
            delta: Delta of the loss with respect to the outputs of the layer.
        Returns:
            The corresponding delta of the layer (d_cost/d_z).
        """
        return np.reshape(delta, self.input_shape)

    def summary(self) -> str:
        return f"{self.name} ({self.input_shape} -> {self.output_shape})"

    def set_weights(self, weights: np.ndarray = None, bias: np.ndarray = None):
        raise NotImplementedError("Flatten layer has no weights")

    def update(self, optimizer, delta: np.ndarray):
        pass

    def count_params(self) -> int:
        return 0
