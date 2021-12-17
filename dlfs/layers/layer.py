from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from dlfs.optimizers import Optimizer
from dlfs.activation_functions import ActivationFunction, get_activation_function


class Layer(ABC):
    """
    Base class for all layers.

    Args:
        input_shape (tuple or None): Shape of the input.
        output_shape (tuple): Shape of the output.
        activation_function (str): Activation function to use.
        name (str): Name of the layer.
        trainable (bool): Whether the layer is trainable.
    """

    def __init__(self,
                 input_shape: Optional[tuple] = None,
                 output_shape: Optional[tuple] = None,
                 activation: Optional[str] = None,
                 name: str = None,
                 trainable: bool = True):

        self.__input_shape = input_shape if input_shape else None
        self.__output_shape = output_shape
        self.__name = name
        self.__trainable = trainable
        self.__initialized = False
        self.__optimizer = None
        self.__activation = get_activation_function(activation)

    # Getters
    # -------------------------------------------------------------------------

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

    @property
    def initialized(self) -> bool:
        return self.__initialized

    @property
    def optimizer(self) -> Optimizer:
        return self.__optimizer

    @property
    def activation(self) -> ActivationFunction:
        return self.__activation

    # Setters
    # -------------------------------------------------------------------------
    @input_shape.setter
    def input_shape(self, input_shape: tuple):
        self.__input_shape = input_shape

    @output_shape.setter
    def output_shape(self, output_shape: tuple):
        self.__output_shape = output_shape

    @trainable.setter
    def trainable(self, trainable: bool):
        self.__trainable = trainable

    @initialized.setter
    def initialized(self, initialized: bool):
        self.__initialized = initialized

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{optimizer} is not an instance of Optimizer")

        self.__optimizer = optimizer

    @activation.setter
    def activation(self, activation: str):
        self.__activation = get_activation_function(activation)

    # Abstract methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def initialize(self, input_shape: tuple):
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass of the layer.
        Args:
            inputs: input to the layer.
            training: whether the layer is in training mode.
        Returns:
            output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer.
        Args:
            gradients: gradients of the layer.
        Returns:
            gradients of the input.
        """
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def update(self, gradients: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def count_params(self) -> int:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.__name})"
