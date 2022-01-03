import numpy as np

from .layer import Layer


class Flatten(Layer):
    """
    Flatten layer. It flattens the input to a 1D vector.
    """

    def __init__(self, name: str = "Flatten"):
        super(Flatten, self).__init__(name=name, has_weights=False)

    def initialize(self, input_shape: tuple):
        """
        Initialize the layer.
        Args:
            input_shape: The input shape.
        """
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.initialized = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass of the layer.
        Args:
            x: Input to the layer.
            training: For compatibility with the base class.
        Returns:
            A 1D vector.
        """
        return np.reshape(x, self.output_shape)

    def get_delta(self, last_delta: np.ndarray, dz_da: np.ndarray) -> np.ndarray:
        """
        Calculates the delta of the layer based on the delta of the next layer and derivative of the output of this
        layer (i) with respect to the z of the next layer (i+1).
        Args:
            last_delta: delta of the next layer.
            dz_da: derivative of the output of this layer (i) with respect to the z of the next layer (i+1). The
                expected value od dz_da here is W.T assuming that the next layer is a dense layer.
        Returns:
            The corresponding delta of the layer (d_cost/d_z).
        """
        delta = last_delta * dz_da if last_delta.shape == self.output_shape else last_delta @ dz_da
        return np.reshape(delta, self.input_shape)

    def summary(self) -> str:
        return f"{self.name} ({self.input_shape} -> {self.output_shape})"

    def set_weights(self, weights: np.ndarray = None, bias: np.ndarray = None):
        raise NotImplementedError("Flatten layer has no weights")

    def get_dz_da(self) -> np.ndarray:
        return np.ones(self.output_shape)

    def update(self, gradients: np.ndarray):
        pass

    def count_params(self) -> int:
        return 0
