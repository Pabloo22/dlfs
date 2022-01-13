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
"""Contains functions related to the Activation Functions."""

from typing import Union

from . import ActivationFunction, Sigmoid, ReLU, Softmax, Linear, Tanh


def get_activation_function(activation_function_name: Union[str, None]) -> Union[ActivationFunction, None]:
    """Returns an activation function object or None if there is no activation function (the same as using 'Linear')

    Args:
        activation_function_name: The name of the activation function. The possible names are:
            - "relu": ReLU
            - "softmax": Softmax
            - "sigmoid": Sigmoid
            - "tanh": Tanh
            - "linear": Linear
            - None: No activation function (i.e. the input is passed through, the same as Linear)
    """
    if activation_function_name is None:
        return None

    if activation_function_name == 'relu':
        return ReLU()
    elif activation_function_name == 'softmax':
        return Softmax()
    elif activation_function_name == 'sigmoid':
        return Sigmoid()
    elif activation_function_name == 'linear':
        return Linear()
    elif activation_function_name == 'tanh':
        return Tanh()
    else:
        raise ValueError(f'Unknown activation function name: {activation_function_name}')
