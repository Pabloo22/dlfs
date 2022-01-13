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
"""Contains the get_optimizer function."""

from . import Optimizer, SGD, SGDMomentum


def get_optimizer(optimizer_name: str) -> Optimizer:
    """Get an optimizer by name.

    Args:
        optimizer_name (str): The name of the optimizer. Valid names are:
            - sgd
            - sgd_momentum
            - adam
            - adagrad

    Returns:
        The optimizer.
    """
    if optimizer_name == "sgd":
        return SGD()
    elif optimizer_name == "sgd_momentum":
        return SGDMomentum()
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")
