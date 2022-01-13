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
"""Contains the WinogradConvolutioner class"""

import numpy as np
from typing import Union, Tuple, List
import tensorly
from skimage.util import view_as_blocks
from dlfs.convolutioners import Convolutioner


class WinogradConvolutioner(Convolutioner):
    """A convolutioner that uses the Winograd convolution algorithm.

    The Winograd algorithm reduces the number of multiplication operations by transforming the input feature map and
    executing a series of transformations on the filter. The result is a much faster convolution.

    This class is not ready for use yet.
    """

    def __init__(self,
                 image_size: Union[int, tuple],
                 kernel_size: Union[int, tuple],
                 padding: Union[int, tuple] = (0, 0),
                 stride: Union[int, tuple] = (1, 1),
                 blocksize: Tuple[int, int] = None,
                 data_format: str = 'channels_last'):

        super().__init__(image_size, kernel_size, padding, stride, data_format=data_format)

        if blocksize is None:
            self.blocksize = None
            self.set_block_size()
        else:
            self.blocksize = blocksize if isinstance(blocksize, tuple) else (blocksize, blocksize)
        self.__transformed_images = []
        self.__transformed_filter = None
        self.block_output_size = None
        self.get_output_shape()
        self.__y_t_matrices = []
        self.__image_transformers = []  # X
        self.__filter_transformers = []  # W
        self.__get_matrices()  # List of arrays
        self.batch_count = 0

    @staticmethod
    def convolve_multichannel(image: np.ndarray,
                              kernel: np.ndarray,
                              stride: Union[int, tuple] = (1, 1),
                              blocksize: Tuple[int, int] = None,
                              transformed_image: np.ndarray = None,
                              y_t_matrices: List[np.ndarray] = None,
                              image_transformers=None) -> np.ndarray:
        out = np.array([tensorly.tenalg.multi_mode_dot(i * kernel, y_t_matrices, list(range(3))) for i in transformed_image])
        print(out)

    @staticmethod
    def convolve_grayscale(image: np.ndarray,
                           kernel: np.ndarray,
                           stride: Union[int, tuple] = (1, 1),
                           blocksize: Tuple[int, int] = None,
                           transformed_image: np.ndarray = None,
                           y_t_matrix=None,
                           image_transformer=None) -> np.ndarray:
        out = np.array([y_t_matrix[0] @ i @ y_t_matrix[1].T for i in transformed_image])
        print(out)

    def __get_matrices(self):
        calculated = []
        for i, j in zip(self.block_output_size, self.image_size):
            if (i, j) in calculated:
                index = calculated.index((i, j))
                self.__y_t_matrices.append(self.__y_t_matrices[index])
                self.__image_transformers.append(self.__image_transformers[index])
                self.__filter_transformers.append(self.__filter_transformers[index])
            else:
                yt, x, w = WinogradConvolutioner.winograd_get_matrices(i, j)
                self.__y_t_matrices.append(yt)
                self.__image_transformers.append(x)
                self.__filter_transformers.append(w)
                calculated.append((i, j))

    def get_output_shape(self) -> Tuple[int, int]:
        """Returns the output shape of the convolution."""
        '''if not len(self.image_size) == len(self.kernel_size):
            raise ValueError('Image and kernel must have the same number of dimensions.')'''

        self.block_output_size = np.array([self.blocksize[i] + self.kernel_size[i] - 1 for i in range(len(self.blocksize))])
        num_of_blocks = np.array((self.image_size[1] / self.block_output_size[0], self.image_size[2] / self.block_output_size[1]))

        if not float(np.prod(num_of_blocks)).is_integer():
            raise ValueError(f'The image size must be divisible by the blocksize. {self.image_size} is not divisible '
                             f'by {self.block_output_size}.')

        output_size = self.block_output_size * num_of_blocks
        return tuple(output_size)

    @staticmethod
    def __transform_multichannel(block: np.ndarray, x: List[np.ndarray]):
        """Returns the image transform for a multichannel image.

        Args:
            block: The image to transform.
            x: The image transform matrices.
        """

        # if block.ndim == 2:
        #     return x[0].T @ block @ x[1]
        # elif block.ndim == 3:
        #     return tensorly.tenalg.multi_mode_dot(block, x, list(range(block.ndim)))

        print(block.shape)
        return np.array([tensorly.tenalg.multi_mode_dot(block[i, j, 0], x, list(range(3))) for j in range(block.shape[1])
                         for i in range(block.shape[0])])

    @staticmethod
    def __transform_grayscale(block: np.ndarray, x: List[np.ndarray]):
        """Returns the image transform for a multichannel image.

        Args:
            block: The image to transform.
            x: The image transform matrices.
        """

        # if block.ndim == 2:
        #     return x[0].T @ block @ x[1]
        # elif block.ndim == 3:
        #     return tensorly.tenalg.multi_mode_dot(block, x, list(range(block.ndim)))

        return np.array([x[0].T @ block[i, j] @ x[1] for j in range(block.shape[1]) for i in range(block.shape[0])])

    @staticmethod
    def __transform_filter(kernel: np.ndarray, m: List[np.ndarray]):
        """Returns the image transform for a multichannel image.

        Args:
            kernel: The filter to transform.
            m: The image transform matrices.
        """

        if kernel.ndim == 2:
            return m[0] @ kernel @ m[1].T
        elif kernel.ndim == 3:
            return tensorly.tenalg.multi_mode_dot(kernel, m, list(range(kernel.ndim)))

    def set_block_size(self):
        num_of_passes = (self.image_size[1] - self.kernel_size[0], self.image_size[2] - self.kernel_size[1])
        shape1 = -1
        shape2 = -1
        print(self.image_size)
        for i in range(self.kernel_size[0] + 1, self.image_size[1]):
            if num_of_passes[0] % i == 0:
                shape1 = i
                print(i)
                break
        for i in range(self.kernel_size[1] + 1, self.image_size[2]):
            if num_of_passes[1] % i == 0:
                shape2 = i
                print(num_of_passes, i)
                break
        self.blocksize = np.array((shape1, shape2))

    def convolve(self,
                 x: np.ndarray,
                 kernel: np.ndarray,
                 using_batches: bool = True,
                 data_format: str = 'channels_last',
                 **kwargs) -> np.ndarray:
        """Returns the convolution of the image and kernel.

        Args:
            x: The image to convolve.
            kernel: The kernel to convolve with.
            using_batches: Whether or not x is a batch of images.
            data_format: The data format of the image.
            **kwargs: Additional keyword arguments. Allowed keywords are:
                - 'batch_count': Used to know whether or not a preprocessing step is needed. If the batch count is
                    different from the last batch count or if it is not provided, we are dealing with a new batch,
                    so we need to preprocess the images and kernels.

        """

        # Add padding to the image if necessary
        if self.padding != (0, 0):
            x = self.pad_image(x, self.padding, data_format=data_format, using_batches=using_batches)

            self.__transformed_filter = WinogradConvolutioner.__transform_filter(kernel, self.__filter_transformers)

        if using_batches:
            # Move batch dimension to the last dimension
            x = np.moveaxis(x, 0, -1)

            if x.ndim == 4:

                # currently only supports data_format = 'channels_last'. So...
                if data_format != 'channels_last':
                    x = np.moveaxis(x, 0, 2)

                # get the blocks
                n_channels = x.shape[2]  # The only data format allowed is 'channels_last'

                # blocks contains 'batch_size' number of numpy arrays each one containing the
                # blocks of the image (views)
                print(x[...,0].shape, (*self.block_output_size, n_channels))
                blocks = [view_as_blocks(image, (*self.block_output_size, n_channels)) for image in x]

                if self.batch_count != kwargs.get('batch_count', None):
                    # blocks contains 'batch_size' number of numpy arrays each one containing the
                    # blocks of the image (views)
                    self.blocks = [view_as_blocks(x[..., i], (*self.blocksize, n_channels)) for i in range(x.shape[-1])]
                    self.batch_count = kwargs['batch_count']
                    self.__transformed_images = [
                        WinogradConvolutioner.__transform_multichannel(block, self.__image_transformers,)
                        for block in blocks]

                return np.array([self.convolve_multichannel(x[i],
                                                            self.__transformed_filter,
                                                            self.stride,
                                                            **kwargs,
                                                            transformed_image=self.__transformed_images[i],
                                                            y_t_matrices=self.__y_t_matrices)
                                 for i in range(x.shape[0])])
            elif x.ndim == 3:

                # get the blocks
                blocks = [view_as_blocks(image, tuple(self.block_output_size)) for image in x]
                self.__transformed_images = [WinogradConvolutioner.__transform_grayscale(block,
                                                                                            self.__image_transformers)
                                             for block in blocks]
                return np.array([self.convolve_grayscale(image, self.__transformed_filter, self.stride, **kwargs,
                                                         transformed_image=self.__transform_multichannel(x,
                                                                                                         self.__filter_transformers),
                                                         y_t_matrix=self.__y_t_matrices,
                                                         image_transformer=self.__image_transformers) for image in x])
        else:
            if x.ndim == 2:
                return self.convolve_grayscale(x, kernel, self.stride, **kwargs,
                                               transformed_image=self.__transform_multichannel(x,
                                                                                               self.__filter_transformers),
                                               y_t_matrix=self.__y_t_matrices,
                                               image_transformer=self.__image_transformers)
            elif x.ndim == 3:
                return self.convolve_multichannel(x, kernel, self.stride, **kwargs,
                                                  transformed_image=
                                                  self.__transform_multichannel(x, self.__filter_transformers),
                                                  y_t_matrices=self.__y_t_matrices,
                                                  image_transformers=self.__image_transformers)

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
        """Returns a list of homogeneous coordinates of size num_points

        This function generates homogeneous coordinates, starting from (0, 1), then, (1/2, 1), then (-1/2, 1),
        and the infinite point (0, 1).

            [(f_0, g_0), ..., (f_a-1, g_a-1)]

        Args:
            num_points (int): The desired number of points
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
