from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from dlfs.activation_functions import ActivationFunction, get_activation_function


class Layer(ABC):
    """
    Base class for all layers.

    Args:
        input_shape (tuple or None): Shape of the input.
        output_shape (tuple): Shape of the output.
        activation (str): Activation function to use.
        name (str): Name of the layer.
        trainable (bool): Whether the layer is trainable.
    """

    def __init__(self,
                 input_shape: Optional[tuple] = None,
                 output_shape: Optional[tuple] = None,
                 activation: Optional[str] = None,
                 name: str = None,
                 trainable: bool = True,
                 has_weights: bool = True):

        self.input_shape = input_shape if input_shape else None
        self.output_shape = output_shape
        self.name = name
        self.weights = None
        self.bias = None
        self.__has_weights = has_weights  # Whether the layer has weights and biases.
        self.trainable = trainable
        self.initialized = False
        self.__activation = get_activation_function(activation)
        self.inputs = None
        self.z = None  # output of the layer before activation

    # Getters
    # -------------------------------------------------------------------------

    @property
    def activation(self) -> ActivationFunction:
        return self.__activation

    @property
    def has_weights(self) -> bool:
        return self.__has_weights

    # Setters
    # -------------------------------------------------------------------------

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

    @abstractmethod
    def set_weights(self, weights: np.ndarray = None, bias: np.ndarray = None):
        raise NotImplementedError

    @abstractmethod
    def get_delta(self, last_delta: np.ndarray, dz_da: np.ndarray) -> np.ndarray:
        """
        Calculates the delta of the layer based on the delta of the next layer and derivative of the output of this
        layer (i) with respect to the z of the next layer (i+1).
        Args:
            last_delta: delta of the next layer.
            dz_da: derivative of the output of this layer (i) with respect to the z of the next layer (i+1).
        Returns:
            The corresponding delta of the layer (d_cost/d_z).
        """

    @abstractmethod
    def get_dz_da(self) -> np.ndarray:
        """
        Returns:
            The derivative of the output of this layer (i) with respect to the z of the next layer (i+1).
        """

    @abstractmethod
    def summary(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def update(self, optimizer, gradients: np.ndarray):
        """
        Updates the weights and biases of the layer.
        Args:
            optimizer (Optimizer): optimizer to use.
            gradients: gradients of the cost function with respect to the weights and biases of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def count_params(self) -> int:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
