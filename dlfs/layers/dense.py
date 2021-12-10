import numpy as np

from .layer import Layer
from dlfs.activation_functions import get_activation_function
from dlfs.optimizers import Optimizer


class Dense(Layer):
    """
    Dense layer class

    Args:
        n_neurons (int): number of neurons
        activation (str): activation function
        name (str): layer name
    """

    def __init__(self, n_neurons: int, activation: str = None, name: str = "Dense", input_shape: tuple = None):

        super(Dense, self).__init__(input_shape=input_shape, output_shape=(n_neurons,), name=name)
        self.__n_neurons = n_neurons
        self.__activation = get_activation_function(activation) if activation else None
        self.weights = None
        self.bias = np.zeros((1, n_neurons))
        self.inputs = None
        self.outputs = None

    # Getters
    # ----------------------------------------------------------------------------------------------------

    @property
    def n_neurons(self):
        return self.__n_neurons

    @property
    def activation(self):
        return self.__activation

    # Setters
    # ----------------------------------------------------------------------------------------------------

    @activation.setter
    def activation(self, activation_name: str):
        self.__activation = get_activation_function(activation_name)

    # Methods
    # ----------------------------------------------------------------------------------------------------
    def initialize(self, input_shape: tuple, optimizer: Optimizer):
        """
        Initialize the layer. Should be called after the input shape is set.

        Args:
            input_shape (tuple): input shape of the layer, it has the form (n_samples, n_features)
            optimizer (Optimizer): optimizer used to optimize the layer
        """
        self.input_shape = input_shape
        # we use Xavier initialization [https://www.deeplearning.ai/ai-notes/initialization/]
        # the weights has the shape (n_features, n_neurons)
        # For instance, if the number of features is 3 and the number of neurons is 5,
        # each column of the weights matrix represents a neuron and the values of the column are its weights
        # of the neuron.
        # the biases has the shape (1, n_neurons)
        self.weights = np.random.randn(input_shape[1], self.__n_neurons) * np.sqrt(1 / input_shape[1])
        self.bias = np.zeros((1, self.__n_neurons))
        self.initialized = True

        self.optimizer = optimizer

    def forward(self, inputs, training: bool = False) -> np.ndarray:
        """
        Forward pass of the layer.

        Args:
            inputs (np.ndarray): inputs of the layer
            training (bool): for compatibility with other layers

        Returns:
            np.ndarray: outputs of the layer
        """
        self.inputs = inputs
        # inputs has the shape (n_samples, n_features) and weights has the shape (n_features, n_neurons)
        self.outputs = np.dot(inputs, self.weights) + self.bias
        if self.__activation:
            self.outputs = self.__activation.forward(self.outputs)
        return self.outputs

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer.

        Args:
            gradients (np.ndarray): gradients of the next layer

        Returns:
            np.ndarray: gradients of the current layer
        """
        if self.__activation:
            gradients = self.__activation.gradient(self.outputs) * gradients
        gradients = np.dot(gradients, self.weights.T)
        return gradients

    def update(self, gradients: np.ndarray):
        """
        Update the weights and biases of the layer.

        Args:
            gradients (np.ndarray): gradients of the current layer
        """

        if self.weights is None or self.bias is None or self.optimizer is None:
            raise ValueError("The layer has not been initialized")

        params: tuple = (self.weights, self.bias)

        self.optimizer.update(params, gradients)

    def summary(self) -> str:
        """
        Summary of the layer.

        Returns:
            str: summary of the layer
        """
        return f"Dense: {self.__n_neurons} neurons\t output_shape={self.output_shape}\t " \
               f"n_params={self.weights.size + self.bias.size}"
