import numpy as np

from .layer import Layer
from dlfs.optimizers import Optimizer


class Dropout(Layer):

    def __init__(self, p: float = 0.5, name: str = "Dropout"):
        super(Dropout, self).__init__(name=name, has_weights=False)
        self.prob_keep = 1 - p  # probability of keeping a neuron active
        self.mask = None

    def initialize(self, input_shape: tuple):
        """
        Initialize input shape and output shape.
        Args:
            input_shape: shape of input data
        """
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.initialized = True

    def forward(self, x: np.ndarray, training: bool = True):
        """
        Forward pass of the layer.
        Args:
            x: input to the layer.
            training: whether the layer is in training mode.
        Returns:
            output of the layer.
        """
        if training:
            self.mask = np.random.binomial(1, self.prob_keep, x.shape) / self.prob_keep
            return x * self.mask
        else:
            return x

    def get_d_inputs(self, delta: np.ndarray) -> np.ndarray:
        """
        Returns the derivative of the cost function with respect to the input of the layer.
        Args:
            delta: derivative of the cost function with respect to the output of the layer.
        Returns:
            derivative of the cost function with respect to the input of the layer.
        """
        return delta * self.mask

    def summary(self):
        return f"{self.name} (p={1 - self.prob_keep})"

    def update(self, optimizer: Optimizer, gradients: np.ndarray):
        """
        Implemented for compatibility with the Layer interface.
        """
        pass

    def count_params(self) -> int:
        return 0

    def set_weights(self, weights: np.ndarray = None, bias: np.ndarray = None):
        raise NotImplementedError("Dropout layer has no weights")
