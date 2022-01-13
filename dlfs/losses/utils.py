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
"""Contains the get_loss function."""

from . import MAE, MSE, BinaryCrossentropy, CategoricalCrossentropy, LossFunction


def get_loss_function(loss_name: str) -> LossFunction:
    """Returns the loss function corresponding to the loss name.

    Raises:
        ValueError: If the loss name is not recognized.
    """

    if loss_name == 'mae':
        return MAE()
    elif loss_name == 'mse':
        return MSE()
    elif loss_name == 'binary_crossentropy':
        return BinaryCrossentropy()
    elif loss_name == 'categorical_crossentropy':
        return CategoricalCrossentropy()
    else:
        raise ValueError(f'Unknown loss function: {loss_name}')
