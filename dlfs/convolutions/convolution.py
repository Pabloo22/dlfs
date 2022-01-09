from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class Convolution(ABC):

    def __init__(self,
                 image_size: Union[int, tuple],
                 kernel_size: Union[int, tuple],
                 padding: bool = False,
                 stride: Union[int, tuple] = (1, 1)):

        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding
        self.stride = (stride, stride) if isinstance(stride, int) else stride

    @staticmethod
    @abstractmethod
    def convolve_grayscale(image: np.ndarray,
                           kernel: np.ndarray,
                           padding: bool = False,
                           stride: Union[int, tuple] = (1, 1),
                           using_batches: bool = False) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def convolve_multichannel(image: np.ndarray,
                              kernel: np.ndarray,
                              padding: bool = False,
                              stride: Union[int, tuple] = (1, 1),
                              using_batches: bool = False) -> np.ndarray:
        pass

    def convolve(self,
                 image: np.ndarray,
                 kernel: np.ndarray,
                 using_batches: bool = True) -> np.ndarray:

        if using_batches:
            if image.ndim == 4:
                return self.convolve_multichannel(image, kernel, self.padding, self.stride, using_batches)
            elif image.ndim == 3:
                return self.convolve_grayscale(image, kernel, self.padding, self.stride, using_batches)
        else:
            if image.ndim == 3:
                return self.convolve_grayscale(image, kernel, self.padding, self.stride, using_batches=False)
            elif image.ndim == 4:
                return self.convolve_multichannel(image, kernel, self.padding, self.stride, using_batches=False)

        raise ValueError("Image must be 2D or 3D.")

    @staticmethod
    def pad_image(image: np.ndarray, kernel_size: tuple, using_batches: bool = False) -> np.ndarray:

        kernel_height, kernel_width = kernel_size

        if not using_batches:
            if image.ndim == 3:
                image = np.pad(image, ((0, 0), (kernel_height // 2, kernel_height // 2),
                                       (kernel_width // 2, kernel_width // 2)),
                               mode='constant', constant_values=0.)
            elif image.ndim == 2:
                image = np.pad(image, ((kernel_height // 2, kernel_height // 2),
                                       (kernel_width // 2, kernel_width // 2)),
                               mode='constant', constant_values=0.)
            else:
                raise ValueError("Image must be 2D or 3D.")
        else:
            if image.ndim == 4:
                image = np.pad(image, ((0, 0), (kernel_height // 2, kernel_height // 2),
                                       (kernel_width // 2, kernel_width // 2), (0, 0)),
                               mode='constant', constant_values=0.)
            elif image.ndim == 3:
                image = np.pad(image, ((kernel_height // 2, kernel_height // 2),
                                       (kernel_width // 2, kernel_width // 2), (0, 0)),
                               mode='constant', constant_values=0.)
            else:
                raise ValueError("Image must be 2D or 3D.")

        return image

    @staticmethod
    def extract_blocks(matrix, blocksize, keep_as_view=False):
        m, n = matrix.shape
        b0, b1 = blocksize
        if not keep_as_view:
            return matrix.reshape(m // b0, b0, n // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)
        else:
            return matrix.reshape(m // b0, b0, n // b1, b1).swapaxes(1, 2)

    def __call__(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return self.convolve(image, kernel)

    def __str__(self):
        return f"{self.__class__.__name__}({self.kernel_size}, {self.padding}, {self.stride})"
