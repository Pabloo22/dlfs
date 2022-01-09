from typing import Union

from . import Convolution, SimpleConvolution, WinogradConvolution


def get_convolution(convolution_type: str,
                    image_size: Union[int, tuple],
                    kernel_size: Union[int, tuple],
                    padding: bool = False,
                    stride: Union[int, tuple] = (1, 1)) -> Convolution:
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

    Returns:
        An activation function object or None if there is no activation function (the same as using 'Linear').
    """

    if convolution_type == "simple":
        return SimpleConvolution(image_size, kernel_size, padding, stride)
    elif convolution_type == "winograd":
        return WinogradConvolution(image_size, kernel_size, padding, stride)
    else:
        raise ValueError(f'Unknown activation function name: {convolution_type}')