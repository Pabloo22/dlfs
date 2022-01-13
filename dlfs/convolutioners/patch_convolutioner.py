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
"""Contains the PatchConvolutioner class-."""

import numpy as np
from typing import Union


from .convolutioner import Convolutioner


class PatchConvolutioner(Convolutioner):
    """Convolutioner for patch-based convolutioners.

    This class implements the convolutioner interface for patch-based convolutioners. For avoid using
    for loops (which are slow), we can take advantage of how numpy arrays are stored in memory.

    This class is not ready for use yet.

    Args:
        image_size (tuple[int, int] or int): The size of the image to convolve. If an int is provided,
            it is assumed to be the size of the image along the first two dimensions.
        kernel_size (tuple[int, int] or int): The size of the kernel to convolve. If an int is provided,
            it is assumed to be the size of the kernel along the first two dimensions.
        padding (tuple[int, int] or int): The padding to apply to the image. If an int is provided,
            it is assumed to be the padding along the first two dimensions.
        stride (tuple[int, int] or int): The stride of the convolution. If an int is provided,
            it is assumed to be the stride along the first two dimensions.
        data_format (str): The data format of the image and kernel. Must be either 'channels_first' or 'channels_last'.
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
