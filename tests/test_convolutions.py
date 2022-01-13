import numpy as np

from dlfs.layers import Conv2D
from dlfs.layers import MaxPooling2D
from dlfs.convolutions import SimpleConvolutioner, Convolutioner, WinogradConvolutioner
from skimage.util.shape import view_as_windows


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

    # convert to data format 'channel last'
    img = np.moveaxis(img, 0, -1)
    print(img)

    padded_img = Convolutioner.pad_image(img, padding=1)
    print(padded_img)
    # convert to data format 'channel first' to see if the result is correct
    padded_img = np.moveaxis(padded_img, -1, 0)
    print(padded_img)


def test_conv():
    img = np.array([[2, 2, 1, 3],
                    [0, 3, 2, 1],
                    [1, 1, 0, 2],
                    [2, 0, 0, 1]])

    filter = np.array([[1, 0],
                       [2, -2]])

    convolutioner = SimpleConvolutioner(4, 2, stride=1, padding=0)
    print(convolutioner.convolve(img, filter, data_format="channel_last", using_batches=False))


def test_get_patches():
    img = np.array([[[2, 2, 1, 3],
                     [0, 3, 2, 1],
                     [1, 1, 0, 2],
                     [2, 0, 0, 1]],
                    [[2, 2, 1, 3],
                     [0, 3, 2, 1],
                     [1, 1, 0, 2],
                     [2, 0, 0, 1]]])

    k = np.array([[[1, 0],
                   [2, -2]],
                  [[1, 0],
                   [2, -2]]])

    # convert image to channels last
    img = np.moveaxis(img, 0, -1)  # image shape (4, 4, 2)

    convolutioner = SimpleConvolutioner(img.shape, k.shape, stride=1, padding=0)
    patches = convolutioner.get_patches(img, k.shape, using_batches=False)

    # convert image to channels first
    # convolved_image = np.moveaxis(convolved_image, -1, 0)
    print(patches)
    print(patches.shape)


def load_cifar10():
    import keras.datasets.cifar10 as cifar10

    (x_train, y_train), _ = cifar10.load_data()

    return x_train


def test_conv_multichannel():
    img = np.array([[[2, 2, 1, 3],
                     [0, 3, 2, 1],
                     [1, 1, 0, 2],
                     [2, 0, 0, 1]],
                    [[2, 2, 1, 3],
                     [0, 3, 2, 1],
                     [1, 1, 0, 2],
                     [2, 0, 0, 1]]])

    k = np.array([[[1, 0],
                   [2, -2]],
                  [[1, 0],
                   [2, -2]]])

    # convert image to channels last
    # img = np.moveaxis(img, 0, -1)  # image shape (4, 4, 2)
    print(k)
    print(img.shape)

    convolutioner = SimpleConvolutioner(img.shape, k.shape, stride=1, padding=0)
    convolved_image = convolutioner.convolve(img, k, using_batches=False)

    # convert image to channels first
    # convolved_image = np.moveaxis(convolved_image, -1, 0)
    print(convolved_image)


def test_winograd_conv_multichannel():
    img = np.arange(28 * 28 * 3 * 2).reshape((2, 28, 28, 3))

    k = np.arange(3 * 3 * 3).reshape((3, 3, 3))

    # convert image to channels last
    # img = np.moveaxis(img, 0, -1)  # image shape (4, 4, 2)
    print(k)
    print(img.shape)

    convolutioner = WinogradConvolutioner(img.shape, k.shape, stride=1, padding=0)
    convolved_image = convolutioner.convolve(img, k, using_batches=True)

    # convert image to channels first
    # convolved_image = np.moveaxis(convolved_image, -1, 0)
    print(convolved_image)


def test_winograd_3d():
    # F (3x4x5, 2,2,2), input size (4x5x6), filter size (2x2x2) output size(3x4x5)
    test_image = np.arange(4 * 4).reshape(4, 4)
    test_filter = np.arange(2 * 2).reshape(2, 2)
    print(WinogradVandermonde.winograd_convolution(test_image, test_filter))


def test_maxpooling():
    img = np.array([[2, 2, 1, 3],
                    [0, 3, 2, 1],
                    [1, 1, 0, 2],
                    [2, 0, 0, 1]]).reshape((1, 4, 4, 1))

    maxpool = MaxPooling2D()
    maxpool.initialize(img.shape)
    maxpooled = maxpool.forward(img)
    print(maxpooled)
