import numpy as np


def convolve_2d(image: np.ndarray, kernel: np.ndarray, padding: bool = False, stride: int = 1) -> np.ndarray:
    """
    Performs a valid convolution on an image with a kernel.

    Args:
        image: A grayscale image.
        kernel: A kernel.
        padding: Whether to pad the image.
        stride: convolution stride size.

    Returns:
        A grayscale image.
    """
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Pad the image if padding is True
    if padding:
        image = np.pad(image, ((kernel_height // 2, kernel_height // 2), (kernel_width // 2, kernel_width // 2)),
                       mode='constant', constant_values=0)

    # Create the output image
    output_height = (image_height - kernel_height) // stride + 1
    output_width = (image_width - kernel_width) // stride + 1
    convolved_image = np.zeros((output_height, output_width))

    # Perform the convolution
    for i in range(output_height):
        for j in range(output_width):
            convolved_image[i, j] = np.sum(
                image[i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width] * kernel)

    return convolved_image
