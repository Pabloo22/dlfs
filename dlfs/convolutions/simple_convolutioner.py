import numpy as np
from typing import Union

from dlfs.convolutions import Convolutioner


class SimpleConvolutioner(Convolutioner):
    """
    A simple convolutioner that performs a convolution on a single image.

    Usage:
        img = np.array([[2, 2, 1, 3],
                        [0, 3, 2, 1],
                        [1, 1, 0, 2],
                        [2, 0, 0, 1]])

        k = np.array([[1,  0],
                      [2, -2]])
        conv = SimpleConvolutioner(img.shape, k.shape)
        print(conv.convolve(img, k, using_batches=False))

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
        """
        Performs a valid convolution on an image (with only a channel) with a kernel.

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
        """
        Performs a valid convolution on an image (with multiple channels) with a kernel.

        Args:
            image: A multichannel image.
            kernel: A kernel.
            stride: convolution stride size.
            data_format: The data format of the image. Either 'channels_last' or 'channels_first'.

        Returns:
            A multichannel image.
        """
        data_format = kwargs.get('data_format', 'channels_last')

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
            error = 0
            error += 1

        assert kernel_channels == image_channels, f"The number of channels in the image and kernel must be the same. " \
                                                  f"Image channels: {image_channels}, " \
                                                  f"kernel channels: {kernel_channels}"

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
