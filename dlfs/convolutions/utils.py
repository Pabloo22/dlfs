from typing import Union, Tuple

from . import Convolutioner, SimpleConvolutioner, WinogradConvolutioner


def get_convolution(convolution_type: str,
                    image_size: Union[int, tuple],
                    kernel_size: Union[int, tuple],
                    padding: tuple = (0, 0),
                    stride: Union[int, tuple] = (1, 1),
                    blocksize: Tuple[int, int] = None) -> Convolutioner:
    """
    Returns an activation function object based on the name of the activation function.
    Args:
        convolution_type: The name of the activation function. The possible names are:
            - "simple": Simple convolution.
            - "winograd": Winograd convolution.
        image_size: The size of the matrix.
        kernel_size: The size of the kernel.
        padding: Whether to use padding or not.
        stride: The stride of the convolution.
        blocksize (Tuple[int, int]): the size of the block, only used with Winograd.

    Returns:
        An activation function object or None if there is no activation function (the same as using 'Linear').
    """

    if convolution_type == "simple":
        return SimpleConvolutioner(image_size, kernel_size, padding, stride)
    elif convolution_type == "winograd":
        return WinogradConvolutioner(image_size, kernel_size, padding, stride, blocksize)
    else:
        raise ValueError(f'Unknown activation function name: {convolution_type}')