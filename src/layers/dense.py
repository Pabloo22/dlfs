from base import Layer
import numpy as np
from src.activation_functions.utils import get_activation_function


class Dense(Layer):

    def __init__(self, n_inputs: int, n_neurons: int, activation: object = None):

        super(Dense, self).__init__(input_shape=(n_inputs,), output_shape=(n_neurons,))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))
        self.weights_gradients = np.zeros_like(self.weights)
        self.bias_gradients = np.zeros_like(self.bias)
        self.weights_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias)
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        """
        Forward pass of the layer.
        """
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.bias
        if self.activation:
            self.outputs = self.activation.forward(self.outputs)
        return self.outputs

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer.
        """
        if self.activation:
            gradients = self.activation.gradient(self.outputs) * gradients
        gradients = np.dot(gradients, self.weights.T)
        self.weights_gradients = np.dot(self.inputs.T, gradients)
        self.bias_gradients = np.sum(gradients, axis=0)
        return gradients

    def reset_gradients(self):
        """
        Reset the gradients of the layer.
        """
        self.weights_gradients = np.zeros_like(self.weights)
        self.bias_gradients = np.zeros_like(self.bias)
        self.weights_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias)
