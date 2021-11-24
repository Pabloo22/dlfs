import numpy as np

from neural_network import NeuralNetwork
from layers.dense import Dense
from activation_functions.utils import get_activation_function


class DenseNeuralNetwork(NeuralNetwork):

    def __init__(self, topology: list, activation: str):
        """
        Class constructor
        """
        # Call the parent class constructor
        super().__init__()
        activation = activation.lower()
        activation_function = get_activation_function(activation)
        self.__layers = [Dense(topology[i], topology[i + 1], activation=activation_function) for i in
                         range(len(topology) - 1)]

    @property
    def layers(self):
        """
        Getter for the layers property
        """
        return self.__layers

    def add(self, layer: Dense):
        """
        Add a layer to the network
        """
        self.layers.append(layer)
