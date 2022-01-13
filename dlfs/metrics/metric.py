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
"""Home of the metric base class."""

from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    """Base class for all metrics.

    A metric is a function that takes two arguments, a prediction and a label,
    and returns a scalar value.
    """

    def __init__(self, name):
        self.name = name

    @staticmethod
    @abstractmethod
    def compute_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    def __str__(self):
        return self.name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.compute_metric(y_true, y_pred)
