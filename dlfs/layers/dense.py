import numpy as np

from .layer import Layer
from dlfs.activation_functions import get_activation_function


class Dense(Layer):
    """
    Dense layer class

    Args:
        n_neurons (int): number of neurons
        activation (str): activation function
        name (str): layer name
    """

    def __init__(self, n_neurons: int, activation: str = None, name: str = "Dense"):

        super(Dense, self).__init__(output_shape=(1, n_neurons), name=name)
        self.__n_neurons = n_neurons
        self.__activation = get_activation_function(activation) if activation else None
        self.__weights = None
        self.__bias = np.zeros((1, n_neurons))
        self.__inputs = None
        self.__outputs = None

    # Getters
    # ----------------------------------------------------------------------------------------------------

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
    def initialize(self, input_shape: tuple):
        """
        Initialize the layer. Should be called after the input shape is set.
        """
        self.input_shape = input_shape
        # we use Xavier initialization [https://www.deeplearning.ai/ai-notes/initialization/]
        self.__weights = np.random.randn(self.input_shape[1], self.n_neurons) * np.sqrt(1 / self.input_shape[1])

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

    def summary(self) -> str:
        """
        Summary of the layer.

        Returns:
            str: summary of the layer
        """
        return f"Dense: {self.__n_neurons} neurons\t output_shape={self.output_shape}\t " \
               f"n_params={self.weights.size + self.__bias.size}"
