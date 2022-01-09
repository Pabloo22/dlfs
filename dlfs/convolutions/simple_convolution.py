import numpy as np
from typing import Union

from dlfs.convolutions import Convolution


class SimpleConvolution(Convolution):

    def __init__(self,
                 image_size: Union[int, tuple],
                 kernel_size: Union[int, tuple],
                 padding: tuple = (0, 0),
                 stride: Union[int, tuple] = (1, 1)):

        super().__init__(image_size, kernel_size, padding, stride)

    @staticmethod
    def convolve_grayscale(image: np.ndarray,
                           kernel: np.ndarray,
                           padding: bool = False,
                           stride: int = 1,
                           using_batches: bool = False) -> np.ndarray:
        """
        Performs a valid convolution on an image (with only a channel) with a kernel.

        Args:
            image: A grayscale image.
            kernel: A kernel.
            padding: Whether to pad the image.
            stride: convolution stride size.
            using_batches: Whether to use batches.

        Returns:
            A grayscale image.
        """
        if using_batches:
            return np.array([SimpleConvolution.convolve_grayscale(image_batch, kernel, padding, stride)
                             for image_batch in image])

        # Get the dimensions of the image and kernel
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # Pad the image if padding is enabled
        if padding:
            image = Convolution.pad_image(image, kernel.shape)

        # Create the output image
        output_height = (image_height - kernel_height) // stride + 1
        output_width = (image_width - kernel_width) // stride + 1
        convolved_image = np.zeros((output_height, output_width))

        # Perform the convolution
        for i in range(output_height):
            for j in range(output_width):
                convolved_image[i, j] = np.sum(image[i * stride:i * stride + kernel_height,
                                               j * stride:j * stride + kernel_width] * kernel)

        return convolved_image

    @staticmethod
    def convolve_multichannel(image: np.ndarray,
                              kernel: np.ndarray,
                              padding: bool = False,
                              stride: Union[int, tuple] = (1, 1),
                              using_batches: bool = False) -> np.ndarray:
        """
        Performs a valid convolution on an image (with multiple channels) with a kernel.

        Args:
            image: A multichannel image.
            kernel: A kernel.
            padding: Whether to pad the image.
            stride: convolution stride size.
            using_batches: Whether to use batches.

        Returns:
            A multichannel image.
        """
        # Pad the image if padding is enabled
        if padding:
            image = Convolution.pad_image(image, kernel.shape, using_batches)

        if using_batches:
            return np.array([SimpleConvolution.convolve_multichannel(image_batch, kernel, padding=False, stride=stride)
                             for image_batch in image])

        # Get the dimensions of the image and kernel
        image_height, image_width, image_channels = image.shape
        kernel_height, kernel_width, kernel_channels = kernel.shape

        # Create the output image
        output_height = (image_height - kernel_height) // stride + 1
        output_width = (image_width - kernel_width) // stride + 1
        convolved_image = np.zeros((output_height, output_width, kernel_channels))

        # Perform the convolution
        for i in range(output_height):
            for j in range(output_width):
                for k in range(kernel_channels):
                    convolved_image[i, j, k] = np.sum(image[i * stride:i * stride + kernel_height,
                                                            j * stride:j * stride + kernel_width,
                                                            k] * kernel[:, :, k])

        return convolved_image
