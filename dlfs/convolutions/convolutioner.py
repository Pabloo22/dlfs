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

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple
from patchify import patchify


class Convolutioner(ABC):
    """
    Abstract class for convolutioners.
    """

    def __init__(self,
                 image_size: Union[int, tuple],
                 kernel_size: Union[int, tuple],
                 padding: Union[int, tuple] = (0, 0),
                 stride: Union[int, tuple] = (1, 1)):

        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding
        self.stride = (stride, stride) if isinstance(stride, int) else stride

    @staticmethod
    @abstractmethod
    def convolve_grayscale(image: np.ndarray,
                           kernel: np.ndarray,
                           padding: Union[int, tuple] = (0, 0),
                           stride: Union[int, tuple] = (1, 1),
                           using_batches: bool = False) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def convolve_multichannel(image: np.ndarray,
                              kernel: np.ndarray,
                              padding: tuple = (0, 0),
                              stride: Union[int, tuple] = (1, 1),
                              using_batches: bool = False) -> np.ndarray:
        pass

    @staticmethod
    def get_patches(image: np.ndarray,
                    patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
                    step: int = 1,
                    using_batches: bool = False) -> np.ndarray:
        """Returns the patches of the image.

        Args:
            image: The image to extract patches from. The image must be a numpy array and must be already
                padded (if necessary).
            patch_size: The size of the patches to extract.
            step: The step size between patches.
            using_batches: Whether to use batches or not.

        Returns:
            The patches of the image.
        """
        if using_batches:
            return np.array([patchify(image[i], patch_size, step) for i in range(image.shape[0])])
        else:
            return patchify(image, patch_size, step)

    def convolve(self,
                 image: np.ndarray,
                 kernel: np.ndarray,
                 using_batches: bool = True) -> np.ndarray:

        if using_batches:
            if image.ndim == 4:
                return self.convolve_multichannel(image, kernel, self.padding, self.stride, using_batches)
            elif image.ndim == 3:
                return self.convolve_grayscale(image, kernel, self.padding, self.stride, using_batches)
        else:
            if image.ndim == 2:
                return self.convolve_grayscale(image, kernel, self.padding, self.stride, using_batches=False)
            elif image.ndim == 3:
                return self.convolve_multichannel(image, kernel, self.padding, self.stride, using_batches=False)

        raise ValueError("Image must be 2D or 3D.")

    @staticmethod
    def pad_image(image: np.ndarray, padding: Union[int, tuple], using_batches: bool = False) -> np.ndarray:

        padding = (padding, padding) if isinstance(padding, int) else padding

        if not using_batches:
            if image.ndim == 3:
                image = np.pad(image, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                               'constant', constant_values=0)
            elif image.ndim == 2:
                image = np.pad(image, ((padding[0], padding[0]), (padding[1], padding[1])),
                               'constant', constant_values=0)
            else:
                raise ValueError("Image must be 2D or 3D.")
        else:
            if image.ndim == 4:
                image = np.pad(image, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)),
                               'constant', constant_values=0)
            elif image.ndim == 3:
                image = np.pad(image, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                               'constant', constant_values=0)
            else:
                raise ValueError("Image must be 2D or 3D.")

        return image

    def __call__(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return self.convolve(image, kernel)

    def __str__(self):
        return f"{self.__class__.__name__}({self.kernel_size}, {self.padding}, {self.stride})"
