import numpy as np
from typing import Union


from dlfs.convolutions import Convolutioner


class WinogradConvolutioner(Convolutioner):

    def __init__(self,
                 image_size: Union[int, tuple],
                 kernel_size: Union[int, tuple],
                 padding: Union[int, tuple] = (0, 0),
                 stride: Union[int, tuple] = (1, 1)):

        super().__init__(image_size, kernel_size, padding, stride)

    @staticmethod
    def convolve_multichannel(image: np.ndarray,
                              kernel: np.ndarray,
                              padding: Union[int, tuple] = (0, 0),
                              stride: Union[int, tuple] = (1, 1),
                              using_batches: bool = False) -> np.ndarray:
        pass

    @staticmethod
    def convolve_grayscale(image: np.ndarray,
                           kernel: np.ndarray,
                           padding: Union[int, tuple] = (0, 0),
                           stride: Union[int, tuple] = (1, 1),
                           using_batches: bool = False) -> np.ndarray:
        pass
