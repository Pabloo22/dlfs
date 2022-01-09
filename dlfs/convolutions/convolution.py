from abc import ABC, abstractmethod
import numpy as np


class Convolution(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def convolve(self) -> np.ndarray:
        pass