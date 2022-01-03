import numpy as np

from dlfs.layers import Conv2D


def add_padding1(image, kernel, padding=True):
    """
    Test the padding of the convolutional layer.
    """
    kernel_height, kernel_width = kernel.shape

    # Pad the image if padding is True
    if padding:
        image = np.pad(image, ((kernel_height // 2, kernel_height // 2), (kernel_width // 2, kernel_width // 2)),
                       mode='constant', constant_values=0)

    return image


def add_padding2(image, kernel, padding=True):
    """
    Test the padding of the convolutional layer. But now we allow to have more than one channel.
    """
    n_channels, kernel_height, kernel_width = kernel.shape

    # Pad the image if padding is True
    if padding:
        image = np.pad(image, ((0, 0),
                               (kernel_height // 2, kernel_height // 2),
                               (kernel_width // 2, kernel_width // 2)),
                       mode='constant', constant_values=0)

    return image


def test_padding1():
    # Test the padding of the convolutional layer
    img = np.array([[1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20]])
    kernel = np.array([[1, 2],
                       [3, 4]])
    print(add_padding1(img, kernel, padding=True))
    print(add_padding1(img, kernel, padding=False))


def test_padding2():

    img = np.array([[[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15],
                     [16, 17, 18, 19, 20]],
                    [[21, 22, 23, 24, 25],
                     [26, 27, 28, 29, 30],
                     [31, 32, 33, 34, 35],
                     [36, 37, 38, 39, 40]]])

    kernel = np.array([[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]]])

    print(add_padding2(img, kernel, padding=True))
    print(add_padding2(img, kernel, padding=False))


if __name__ == '__main__':
    test_padding1()
    test_padding2()
