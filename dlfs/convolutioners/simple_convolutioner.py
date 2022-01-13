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
"""Contains the SimpleConvolutioner class."""

import numpy as np
from typing import Union

from dlfs.convolutioners import Convolutioner


class SimpleConvolutioner(Convolutioner):
    """
    A simple convolutioner that performs a convolution on a single image.

    This convolutioner can be used by Conv2D layers. However, it is not intended to be
    an efficient implementation.

    Usage:
        img = np.array([[2, 2, 1, 3],
                        [0, 3, 2, 1],
                        [1, 1, 0, 2],
                        [2, 0, 0, 1]])

        k = np.array([[1,  0],
                      [2, -2]])
        conv = SimpleConvolutioner(img.shape, k.shape)
        print(conv.convolve(img, k, using_batches=False))

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

        super().__init__(image_size, kernel_size, padding, stride, data_format=data_format)

    @staticmethod
    def convolve_grayscale(image: np.ndarray,
                           kernel: np.ndarray,
                           stride: Union[int, tuple] = (0, 0),
                           data_format: str = 'channels_last',
                           **kwargs) -> np.ndarray:
        """Performs a valid convolution on a grayscale image with a 2D kernel.

        Args:
            image: A grayscale image.
            kernel: A kernel.
            stride: convolution stride size.
            data_format: The data format of the image. Either 'channels_last' or 'channels_first'.

        Returns:
            A grayscale image.
        """

        # Get the dimensions of the image and kernel
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # Create the output image
        stride = (stride, stride) if isinstance(stride, int) else stride
        output_height = (image_height - kernel_height) // stride[0] + 1
        output_width = (image_width - kernel_width) // stride[1] + 1
        convolved_image = np.zeros((output_height, output_width))

        # Perform the convolution
        for i in range(output_height):
            for j in range(output_width):
                convolved_image[i, j] = np.sum(image[i * stride[0]:i * stride[0] + kernel_height,
                                                     j * stride[1]:j * stride[1] + kernel_width] * kernel)
        return convolved_image

    @staticmethod
    def convolve_multichannel(image: np.ndarray,
                              kernel: np.ndarray,
                              stride: Union[int, tuple] = (1, 1),
                              data_format: str = 'channels_last',
                              **kwargs) -> np.ndarray:
        """Performs a valid convolution on an image (with multiple channels) with a 3D kernel.

        Args:
            image: A multichannel image.
            kernel: A kernel.
            stride: convolution stride size.
            data_format: The data format of the image. Either 'channels_last' or 'channels_first'.

        Returns:
            A multichannel image.
        """

        # convert image and kernel to channel first if necessary
        if data_format == 'channels_last':
            image = np.moveaxis(image, -1, 0)
            kernel = np.moveaxis(kernel, -1, 0)

        # Get the dimensions of the image and kernel
        if len(kernel.shape) != 3:
            error = 0
            error += 1

        image_channels, image_height, image_width = image.shape
        kernel_channels, kernel_height, kernel_width = kernel.shape

        if kernel_channels != image_channels:
            raise ValueError('The number of channels in the image and kernel must be the same.'
                             f'Image channels: {image_channels}, kernel channels: {kernel_channels}')

        # Create the output image
        stride = (stride, stride) if isinstance(stride, int) else stride
        output_height = (image_height - kernel_height) // stride[0] + 1
        output_width = (image_width - kernel_width) // stride[1] + 1
        convolved_image = np.zeros((output_height, output_width))

        for y in range(output_height):
            for x in range(output_width):
                convolved_image[y][x] = sum(
                    [np.sum(im[y * stride[0]:y * stride[0] + kernel_height, x * stride[1]:x * stride[1] + kernel_width]
                            * kern) for im, kern in zip(image, kernel)])

        # convert image to channel last if necessary
        if data_format == 'channels_last':
            convolved_image = np.moveaxis(convolved_image, 0, -1)

        return convolved_image
