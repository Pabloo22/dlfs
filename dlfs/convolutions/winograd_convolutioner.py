import numpy as np
from typing import Union, Tuple, List
import tensorly

from dlfs.convolutions import Convolutioner


class WinogradConvolutioner(Convolutioner):
    """A convolutioner that uses the Winograd convolution algorithm.

    The Winograd algorithm reduces the number of multiplication operations by transforming the input feature map and
    executing a series of transformations on the filter. The result is a much faster convolution.
    """

    def __init__(self,
                 image_size: Union[int, tuple],
                 kernel_size: Union[int, tuple],
                 padding: Union[int, tuple] = (0, 0),
                 stride: Union[int, tuple] = (1, 1),
                 blocksize: Tuple[int, int] = None):

        if blocksize is None:
            self.blocksize = None
            self.set_block_size()
        else:
            self.blocksize = blocksize if isinstance(blocksize, tuple) else (blocksize, blocksize)
        self.__image_transform = None
        self.block_output_size = None
        self.get_output_shape()
        self.__yt = []
        self.__x = []
        self.__w = []
        self.__get_matrices()  # List of arrays
        self.image_size = image_size
        self.kernel_size = kernel_size

        super().__init__(image_size, kernel_size, padding, stride)

    @staticmethod
    def convolve_multichannel(image: np.ndarray,
                              kernel: np.ndarray,
                              stride: Union[int, tuple] = (1, 1),
                              blocksize: Tuple[int, int] = None,
                              image_transform: np.ndarray = None,
                              YT=None, X=None) -> np.ndarray:
        pass

    @staticmethod
    def convolve_grayscale(image: np.ndarray,
                           kernel: np.ndarray,
                           stride: Union[int, tuple] = (1, 1),
                           blocksize: Tuple[int, int] = None,
                           image_transform: np.ndarray = None,
                           yt=None, x=None) -> np.ndarray:
        pass

    def __get_matrices(self):
        for i, j in zip(self.block_output_size, self.image_size):
            yt, x, w = WinogradConvolutioner.winograd_get_matrices(i, j)
            self.__yt.append(yt)
            self.__x.append(x)
            self.__w.append(w)

    def get_output_shape(self) -> Tuple[int, int]:
        """Returns the output shape of the convolution.

        Returns:
            The output shape of the convolution.
        """
        if not len(self.image_size) == len(self.kernel_size):
            raise ValueError('')
        already_calculated = []
        winograd_matrices = []
        if len(self.image_size) == 2:
            num_of_blocks = np.array((self.image_size[0] / self.blocksize[0], self.image_size[1] / self.blocksize[1]))
            if np.prod(num_of_blocks).dtype != int:
                raise ValueError
            self.block_output_size = np.array(self.blocksize) + 1 - np.array(self.kernel_size)
            output_size = self.block_output_size * num_of_blocks
            return tuple(output_size)

    @staticmethod
    def __get_image_transform_multichannel(image, x):
        if image.ndim == 2:
            return x[0].T @ image @ x[1]
        elif image.ndim == 3:
            return tensorly.tenalg.multi_mode_dot(image, [_ for _ in x], list(range(image.ndim)))

    def set_block_size(self):
        num_of_passes = (self.image_size[0] - self.kernel_size[0], self.image_size[1] - self.kernel_size[1])
        shape1 = -1
        shape2 = -1
        for i in range(self.image_size[0] + 1, self.image_size[0]):
            if num_of_passes[0] % shape1 == 0:
                shape1 = i
                break
        for i in range(self.image_size[1] + 1, self.image_size[1]):
            if num_of_passes[1] % shape2 == 0:
                shape2 = i
                break
        self.blocksize = np.array((shape1, shape2))

    def convolve(self,
                 x: np.ndarray,
                 kernel: np.ndarray,
                 using_batches: bool = True,
                 **kwargs) -> np.ndarray:

        # Add padding to the image if necessary
        if self.padding != (0, 0):
            x = self.pad_image(x, self.padding, using_batches)

        if using_batches:
            if x.ndim == 4:
                return np.array([self.convolve_multichannel(image, kernel, self.stride, **kwargs,
                                                            image_transform=self.__get_image_transform_multichannel(x,
                                                                                                                    self.__w),
                                                            yt=self.__yt, x=self.__x) for image in x])
            elif x.ndim == 3:
                return np.array([self.convolve_grayscale(image, kernel, self.stride, **kwargs,
                                                         image_transform=self.__get_image_transform_multichannel(x,
                                                                                                                 self.__w),
                                                         yt=self.__yt, x=self.__x) for image in x])
        else:
            if x.ndim == 2:
                return self.convolve_grayscale(x, kernel, self.stride, **kwargs,
                                               image_transform=self.__get_image_transform_multichannel(x, self.__w),
                                               yt=self.__yt, x=self.__x)
            elif x.ndim == 3:
                return self.convolve_multichannel(x, kernel, self.stride, **kwargs,
                                                  image_transform=self.__get_image_transform_multichannel(x, self.__w),
                                                  yt=self.__yt, x=self.__x)

        raise ValueError("Image must be 2D or 3D.")

    @staticmethod
    def vandermonde_matrix(b: int, points: List[Tuple[float, int]]) -> np.ndarray:
        """
        This is equivalent to V_{axb} on the document.
        This function returns a trimmed Vandermonde Matrix of size len(points) image b

            f_0^0 * g_0^b-1       f_0^1 * g_0^b-2       ... f_0^b-1 * g_0^0
            .
            .
            f_a-1^0 * g_a-1^b-1   f_a-1^1 * g_a-1^b-2   ... f_a-1^b-1 * g_a-1^0

        Arguments:
            b (int) : One of the dimensions of the matrix
            points (list[tuple[Union[int, float]]]) : List of homogeneous coordinates
        Returns:
            V (np.ndarray[Union[int, float]]) : The trimmed Vandermonde matrix
        """
        return np.array([[i[0] ** j * i[1] ** (b - j - 1) for j in range(b)] for i in points])

    @staticmethod
    def gen_points(num_points: int) -> List[Tuple[float, int]]:
        """
        This function generates homogeneous coordinates, starting from (0, 1), then, (1/2, 1), then (-1/2, 1), and the infinite point (0, 1).

            [(f_0, g_0), ..., (f_a-1, g_a-1)]

        Args:
            num_points (int): The desired number of points

        Returns:
            list[tuple[Union[int, float]]]: List of homogeneous coordinates
        """
        points = [(0, 1)]
        if num_points % 2:
            points.extend([(y / 2, 1) for x in range(1, num_points // 2) for y in (x, -x)])
            points.extend([(num_points // 2 / 2, 1), (1, 0)])
        else:
            points.extend([(y / 2, 1) for x in range(1, num_points // 2) for y in (x, -x)])
            points.append((1, 0))
        return points

    @staticmethod
    def winograd_get_matrices(m: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This is equivalent to F(m, n) on the paper.
        This function returns the matrices necessary for calculating the Winograd convolution given the dimension of the filter and the dimension of the output. With those, we can infer the size of the input.
        The usage of these matrices depends on the dimension of the convolution. If we denote image as the signal and w as the filter:
        - If it is 1D: F(m, n) = image * w = Y.T @ [(X.T @ image) * (W @ w)]
        - If it is 2D: F(m image i, n image j) = image * w = Y.T @ [(X.T @ image @ X') * (W @ w @ W'.T)] @ Y'
        In the latter case, if m = i and n = j, then Y = Y', X = X', W = W'. If not, then you would have to calculate both F(m, n) to get Y, X, W and F(i, j) to get Y', X', W'.

        Args:
            m (int): The size of the output
            n (int): The size of the filter

        Returns:
            A tuple with the aforementioned Y.T, X, W in this order.
        """
        num_points = m + n - 1
        points = WinogradConvolutioner.gen_points(num_points)
        y = WinogradConvolutioner.vandermonde_matrix(m, points)
        x = np.linalg.inv(WinogradConvolutioner.vandermonde_matrix(num_points, points))
        w = WinogradConvolutioner.vandermonde_matrix(n, points)
        return y.T, x, w
