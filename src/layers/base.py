import numpy as np


class Layer:
    """
    Base class for all layers.

    Args:
        input_shape (tuple): Shape of the input.
        output_shape (tuple): Shape of the output.
        name (str): Name of the layer.
        trainable (bool): Whether the layer is trainable.
    """

    def __init__(self, input_shape: tuple, output_shape: tuple, name: str = None, trainable: bool = True):
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__name = name
        self.__trainable = trainable

    @property
    def input_shape(self) -> tuple:
        return self.__input_shape

    @property
    def output_shape(self) -> tuple:
        return self.__output_shape

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def trainable(self) -> bool:
        return self.__trainable

    @trainable.setter
    def trainable(self, trainable: bool):
        self.__trainable = trainable

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer.
        Args:
            inputs: input to the layer.
        Returns:
            output of the layer.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__name})"
