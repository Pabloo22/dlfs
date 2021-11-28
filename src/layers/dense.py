import numpy as np

from base import Layer
from src.activation_functions.utils import get_activation_function


class Dense(Layer):
    """
    Dense layer class

    Args:
        n_inputs (int): number of inputs
        n_neurons (int): number of neurons
        activation_name (str): activation function
    """

    def __init__(self, n_inputs: int, n_neurons: int, activation_name: str = None):

        super(Dense, self).__init__(input_shape=(n_inputs,), output_shape=(n_neurons,))
        self.__n_inputs = n_inputs
        self.__n_neurons = n_neurons
        self.__activation = get_activation_function(activation_name)
        self.__weights = np.random.randn(n_inputs, n_neurons)
        self.__bias = np.zeros((1, n_neurons))
        self.__inputs = None
        self.__outputs = None

    # Getters
    # ----------------------------------------------------------------------------------------------------

    @property
    def n_inputs(self):
        return self.__n_inputs

    @property
    def n_neurons(self):
        return self.__n_neurons

    @property
    def activation(self):
        return self.__activation

    @property
    def weights(self):
        return self.__weights

    @property
    def biases(self):
        return self.__bias

    @property
    def inputs(self):
        return self.__inputs

    @property
    def outputs(self):
        return self.__outputs

    # Setters
    # ----------------------------------------------------------------------------------------------------

    @activation.setter
    def activation(self, activation_name: str):
        self.__activation = get_activation_function(activation_name)

    @weights.setter
    def weights(self, weights: np.ndarray):
        self.__weights = weights

    @biases.setter
    def biases(self, biases: np.ndarray):
        self.__bias = biases

    @inputs.setter
    def inputs(self, inputs: np.ndarray):
        self.__inputs = inputs

    @outputs.setter
    def outputs(self, outputs: np.ndarray):
        self.__outputs = outputs

    # Methods
    # ----------------------------------------------------------------------------------------------------

    def forward(self, inputs) -> np.ndarray:
        """
        Forward pass of the layer.

        Args:
            inputs (np.ndarray): inputs of the layer

        Returns:
            np.ndarray: outputs of the layer
        """
        self.__inputs = inputs
        self.__outputs = np.dot(inputs, self.__weights) + self.__bias
        if self.__activation:
            self.__outputs = self.__activation.forward(self.__outputs)
        return self.__outputs

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer.

        Args:
            gradients (np.ndarray): gradients of the next layer

        Returns:
            np.ndarray: gradients of the current layer
        """
        if self.__activation:
            gradients = self.__activation.gradient(self.__outputs) * gradients
        gradients = np.dot(gradients, self.__weights.T)
        return gradients

