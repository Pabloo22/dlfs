import numpy as np

from .layer import Layer
from dlfs.optimizers import Optimizer


class Dropout(Layer):
    """Applies Dropout to the input.

    The dropout layer is a regularization technique that is used to prevent overfitting. It randomly sets a fraction
    of the input units to 0 at each update during training time using a mask, and rescales the remaining units by 1/
    (1-p) during training, where p is the dropout rate. Note that this regularization technique only applies to the
    training phase, when using the `fit` or `train_on_batch` methods.

    Args:
        p (float): The dropout rate.
        seed (int): The seed for the random number generator.
        name (str): The name of the layer.

    """

    def __init__(self, p: float, seed: int = None, name: str = "Dropout"):

        # check if p is in range [0, 1]
        if not 0 <= p <= 1:
            raise ValueError("p must be in range [0, 1]")

        super(Dropout, self).__init__(name=name, has_weights=False)

        self.prob_keep = 1 - p
        self.random_state = np.random.RandomState(seed)
        self.mask = None

    def initialize(self, input_shape: tuple):
        """Initialize input shape and output shape.

        Args:
            input_shape: shape of input data
        """
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.initialized = True

    def forward(self, x: np.ndarray, training: bool = True):
        """Forward pass of the layer.

        Args:
            x: input to the layer.
            training: whether the layer is in training mode.

        Returns:
            output of the layer.
        """
        if training:
            self.mask = self.random_state.binomial(n=1, p=self.prob_keep, size=x.shape).astype(np.float32)
            return x * self.mask
        else:
            return x

    def get_d_inputs(self, delta: np.ndarray) -> np.ndarray:
        """Returns the derivative of the cost function with respect to the input of the layer.

        Args:
            delta: derivative of the cost function with respect to the output of the layer.

        Returns:
            derivative of the cost function with respect to the input of the layer.
        """
        return delta * self.mask

    def summary(self):
        """Returns a string summary of the layer."""
        return f"{self.name} (p={1 - self.prob_keep})"

    def update(self, optimizer: Optimizer, gradients: np.ndarray):
        """Implemented for compatibility with the Layer interface."""

    def count_params(self) -> int:
        """Returns the number of trainable parameters in the layer."""
        return 0

    def set_weights(self, weights: np.ndarray = None, bias: np.ndarray = None):
        """Implemented for compatibility with the Layer interface."""
        raise NotImplementedError("Dropout layer has no weights")
