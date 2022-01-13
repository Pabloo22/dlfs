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
    """Abstract class for convolutioners.

    A convolutioner is a class that can be used to convolve a given image with
    a given kernel.

    Attributes:
        image_size: The size of the image to be convolved.
        kernel_size: The size of the kernel to be convolved with.
        padding: The amount of padding to be used.
        stride: The amount of stride to be used.
    """

    def __init__(self,
                 image_size: Union[int, tuple],
                 kernel_size: Union[int, tuple],
                 padding: Union[int, tuple] = (0, 0),
                 stride: Union[int, tuple] = (1, 1),
                 data_format: str = "channels_last"):

        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.data_format = data_format

    @staticmethod
    @abstractmethod
    def convolve_grayscale(image: np.ndarray,
                           kernel: np.ndarray,
                           stride: Union[int, tuple] = (1, 1),
                           **kwargs) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def convolve_multichannel(image: np.ndarray,
                              kernel: np.ndarray,
                              stride: Union[int, tuple] = (1, 1),
                              **kwargs) -> np.ndarray:
        pass

    @staticmethod
    def get_patches(x: np.ndarray,
                    patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
                    step: int = 1,
                    using_batches: bool = False) -> np.ndarray:
        """Returns the patches of the image.

        Args:
            x: The image to extract patches from. The image must be a numpy array and must be already
                padded (if necessary).
            patch_size: The size of the patches to extract.
            step: The step size between patches.
            using_batches: Whether to use batches or not.

        Returns:
            The patches of the image.
        """
        if using_batches:
            return np.array([patchify(image, patch_size, step) for image in x])
        else:
            return patchify(x, patch_size, step)

    def get_output_shape(self) -> Tuple[int, int]:
        """Returns the output shape of the convolution."""
        output_height = (self.image_size[0] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        output_width = (self.image_size[1] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        return output_height, output_width

    def convolve(self,
                 x: np.ndarray,
                 kernel: np.ndarray,
                 using_batches: bool = True,
                 **kwargs) -> np.ndarray:

        # Add padding to the image if necessary
        if self.padding != (0, 0):
            x = self.pad_image(x, self.padding, self.data_format, using_batches)

        if using_batches:
            if x.ndim == 4:
                return np.array([self.convolve_multichannel(image, kernel, self.stride, **kwargs)
                                 for image in x])
            elif x.ndim == 3:
                return np.array([self.convolve_grayscale(image, kernel, self.stride, **kwargs)
                                 for image in x])
        else:
            if x.ndim == 2:
                return self.convolve_grayscale(x, kernel, self.stride, **kwargs)
            elif x.ndim == 3:
                return self.convolve_multichannel(x, kernel, self.stride, **kwargs)

        raise ValueError("Image must be 2D or 3D.")

    @staticmethod
    def pad_image(x: np.ndarray,
                  padding: Union[int, tuple],
                  data_format: str = "channels_last",
                  using_batches: bool = False) -> np.ndarray:

        padding = (padding, padding) if isinstance(padding, int) else padding

        # If data format is channels_last, then we need to convert the image to channels_first

        if not using_batches:
            if x.ndim == 3:  # A single multichannel image
                # If the data format is channels_last, then we need to convert the image to channels_first
                x = np.transpose(x, (2, 0, 1)) if data_format == "channels_last" else x

                x = np.pad(x, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])))

            elif x.ndim == 2:  # A single grayscale image
                x = np.pad(x, ((padding[0], padding[0]), (padding[1], padding[1])))
            else:
                raise ValueError("Image must be 2D or 3D.")
        else:
            if x.ndim == 4:  # A batch of multichannel images

                # If the data format is channels_last, then we need to convert the image to channels_first
                x = np.transpose(x, (0, 3, 1, 2)) if data_format == "channels_last" else x

                # The first dimension is the batch dimension
                x = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))

            elif x.ndim == 3:  # Batch of grayscale images
                x = np.pad(x, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
            else:
                raise ValueError("Image must be 2D or 3D.")

        # reconvert the image to channels_last if necessary
        if not using_batches:
            if x.ndim == 3:  # A single multichannel image
                x = np.transpose(x, (1, 2, 0)) if data_format == "channels_last" else x
        else:
            if x.ndim == 4:  # A batch of multichannel images
                x = np.transpose(x, (0, 2, 3, 1)) if data_format == "channels_last" else x

        return x

    def __call__(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return self.convolve(image, kernel)

    def __str__(self):
        return f"{self.__class__.__name__}({self.kernel_size}, {self.padding}, {self.stride})"
