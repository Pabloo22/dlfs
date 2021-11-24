import numpy as np
from typing import List

from activation_functions.base import ActivationFunction
from layers.base import Layer


class Sequential:

    def __init__(self, layers: List[Layer] = None, name: str = None):
        """
        Initialize a Sequential model
        :param layers: a list of layers
        """
        self.layers = layers
        self.name = name
        self.loss = None
        self.optimizer = None
        self.metrics = None
        self.trainable = True

    def add(self, layer: Layer):
        """
        Add a layer to the model
        Args:
            layer: the layer to add
        """
        self.layers.append(layer)

    def compile(self, optimizer, loss, metrics=None):
        """
        Compile the model

        Args:
            optimizer: the optimizer to use
            loss: the loss function to use
            metrics: the metrics to use
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def summary(self):
        """
        Print the summary of the model as a table
        """
        print(f"{self.name}")
        print("-" * len(self.name))
        print("Layers:")
        for layer in self.layers:
            print(f"{layer.name}")
            print("-" * len(layer.name))
            print(f"{layer.summary()}")
            print("")

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32, verbose: int = 1,
            validation_data: np.ndarray = None, validation_split: float = 0.0, shuffle: bool = True, initial_epoch=0):
        """

        Args:
            x: the input data
            y: the labels
            epochs: the number of epochs to train the model
            batch_size: the batch size
            verbose: the verbosity mode
            validation_data: the validation data
            validation_split: the validation split
            shuffle: whether to shuffle the data
            callbacks: the callbacks to use
            initial_epoch: the initial epoch
        """
        if self.optimizer is None:
            raise ValueError("You must compile the model before training")

        if validation_split > 0:
            if validation_data is None:
                raise ValueError("If validation_split is set, validation_data must be specified")

    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None, steps=None):
        """

        Args:
            x: the input data
            y: the labels
            batch_size: the batch size
            verbose: the verbosity mode
            sample_weight: the sample weights
            steps: the number of steps to evaluate
        """

    def predict(self, x: np.ndarray, batch_size: int = 32, verbose: int = 1):
        """

        Args:
            x: the input data
            batch_size: the batch size
            verbose: the verbosity mode (0 or 1)
        """
