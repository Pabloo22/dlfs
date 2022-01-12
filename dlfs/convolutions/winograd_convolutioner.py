import numpy as np
from typing import Union, Tuple

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
        self.blocksize = blocksize if isinstance(blocksize, tuple) else (blocksize, blocksize)
        self.__image_transform = None
        self.__YT, self.__X, self.__W = self.__get_matrices() # List of arrays
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
                           YT=None, X=None) -> np.ndarray:
        pass

    def __get_matrices(self):
        return None, None, None


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
            if self.blocksize == None:
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
            num_of_blocks = np.array((self.image_size[0] / self.blocksize[0], self.image_size[1] / self.blocksize[1]))
            if np.prod(num_of_blocks).dtype != int:
                raise ValueError
            block_output_size = self.blocksize + 1 - np.array(self.kernel_size)
            output_size = block_output_size * num_of_blocks
            return tuple(output_size)

    @staticmethod
    def __get_image_transform_multichannel(image, W):
        pass

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
                                                            image_transform=self.__get_image_transform_multichannel(x, self.__W),
                                                            YT=self.__YT, X=self.__X)
                                 for image in x])
            elif x.ndim == 3:
                return np.array([self.convolve_grayscale(image, kernel, self.stride, **kwargs,
                                                         image_transform=self.__get_image_transform_multichannel(x, self.__W),
                                                         YT=self.__YT, X=self.__X)
                                 for image in x])
        else:
            if x.ndim == 2:
                return self.convolve_grayscale(x, kernel, self.stride, **kwargs,
                                               image_transform=self.__get_image_transform_multichannel(x, self.__W),
                                               YT=self.__YT, X=self.__X)
            elif x.ndim == 3:
                return self.convolve_multichannel(x, kernel, self.stride, **kwargs,
                                                  image_transform=self.__get_image_transform_multichannel(x, self.__W),
                                                  YT=self.__YT, X=self.__X)

        raise ValueError("Image must be 2D or 3D.")

