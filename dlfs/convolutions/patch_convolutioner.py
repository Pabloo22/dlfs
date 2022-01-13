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
"""Contains the PatchConvolutioner class"""

import numpy as np
from typing import Union


from .convolutioner import Convolutioner


class PatchConvolutioner(Convolutioner):
    """Convolutioner for patch-based convolutions.

    This class implements the convolutioner interface for patch-based convolutions. For avoid using
    for loops (which are slow), we can take advantage of how numpy arrays are stored in memory.
    """
    def __init__(self,
                 image_size: Union[int, tuple],
                 kernel_size: Union[int, tuple],
                 padding: Union[int, tuple] = (0, 0),
                 stride: Union[int, tuple] = (1, 1),
                 data_format: str = 'channels_last'):

        super().__init__(image_size, kernel_size, padding, stride, data_format)

    @staticmethod
    def convolve_grayscale(image: np.ndarray, kernel: np.ndarray, padding: Union[int, tuple] = (0, 0),
                           stride: Union[int, tuple] = (1, 1), using_batches: bool = False) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def convolve_multichannel(image: np.ndarray, kernel: np.ndarray, padding: tuple = (0, 0),
                              stride: Union[int, tuple] = (1, 1), using_batches: bool = False) -> np.ndarray:
        pass
