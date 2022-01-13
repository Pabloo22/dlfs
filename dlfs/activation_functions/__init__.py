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
"""dlfs activation functions API."""

from .activation_function import ActivationFunction
from .sigmoid import Sigmoid
from .softmax import Softmax
from .relu import ReLU
from .linear import Linear
from .tanh import Tanh
from .utils import get_activation_function
