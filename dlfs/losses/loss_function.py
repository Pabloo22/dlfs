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
"""Home of the Loss base class."""

from abc import ABC, abstractmethod
from numpy import ndarray


class LossFunction(ABC):
    """Base class for loss functions.

    Loss functions are used to compute the loss between the prediction and the
    target. They can also be used as metrics.
    """

    def __init__(self, name):
        self.name = name

    @staticmethod
    @abstractmethod
    def compute_loss(y_true, y_pred) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def gradient(y_true, y_pred) -> ndarray:
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        return self.compute_loss(y_true, y_pred)
