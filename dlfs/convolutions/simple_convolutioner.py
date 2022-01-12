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
                 stride: Union[int, tuple] = (1, 1)):

        super().__init__(image_size, kernel_size, padding, stride)

    @staticmethod
    def convolve_grayscale(image: np.ndarray,
                           kernel: np.ndarray,
                           padding: Union[int, tuple] = (0, 0),
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

        # Pad the image if padding is enabled
        padding = (padding, padding) if isinstance(padding, int) else padding
        image = Convolutioner.pad_image(image, padding) if padding != (0, 0) else image
        if using_batches:
            return np.array([SimpleConvolutioner.convolve_grayscale(image_batch, kernel, 0, stride)
                             for image_batch in image])

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
                              padding: Union[int, tuple] = (0, 0),
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
        padding = (padding, padding) if isinstance(padding, int) else padding
        image = Convolutioner.pad_image(image, padding) if padding != (0, 0) else image

        if using_batches:
            return np.array([SimpleConvolutioner.convolve_multichannel(image_batch, kernel, padding=0, stride=stride)
                             for image_batch in image])

        # Get the dimensions of the image and kernel
        image_height, image_width, image_channels = image.shape
        kernel_height, kernel_width, kernel_channels = kernel.shape

        # Create the output image
        stride = (stride, stride) if isinstance(stride, int) else stride
        output_height = (image_height - kernel_height) // stride[0] + 1
        output_width = (image_width - kernel_width) // stride[1] + 1
        convolved_image = np.zeros((output_height, output_width, kernel_channels))

        # Perform the convolution
        for i in range(output_height):
            for j in range(output_width):
                for k in range(kernel_channels):
                    convolved_image[i, j, k] = np.sum(image[i * stride[0]:i * stride[0] + kernel_height,
                                                            j * stride[1]:j * stride[1] + kernel_width,
                                                            k] * kernel[:, :, k])

        return convolved_image
