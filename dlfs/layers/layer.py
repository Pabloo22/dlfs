from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from dlfs.activation_functions import ActivationFunction, get_activation_function


class Layer(ABC):
    """Base class for all layers.

    Every layer should inherit from this class and implement the abstract methods. A layer is a part of a neural network
    that performs some computation on the input data and produces an output. The output of a layer is the input of the
    next layer. Some layers have weights and biases, which are used to compute the output. These parameters are
    updated during training using backpropagation. This is the reason why there are methods to compute the gradient of
    the loss function with respect to the input of the layer, the weights and the biases. The gradient of the loss
    with respect to the weights and biases is computed in the `update` method.

    Args:
        input_shape (tuple or None): Shape of the input.
        output_shape (tuple): Shape of the output.
        activation (str): Activation function to use.
        name (str): Name of the layer.
        trainable (bool): Whether the layer is trainable.
        has_weights (bool): Whether the layer has weights.

    Attributes:
        input_shape (tuple or None): Shape of the input.
        output_shape (tuple): Shape of the output.
        name (str): Name of the layer. The name is used to identify the layer in the network
            and is used by the optimizer to identify the layer's parameters.
        __weights (np.ndarray): Weights of the layer (None by default). If the layer has weights,
            they can be set using the set_weights method or in the initialize method.
        __bias (np.ndarray): Bias of the layer (None by default). They are set in the
            initialize method or in the set_weights method.
        has_weights (bool): Whether the layer has weights.
        trainable (bool): Whether the layer is trainable.
        initialized (bool): Whether the layer has been initialized.
        activation (ActivationFunction): Activation function to use.
        inputs (np.ndarray): Inputs of the layer. Needed in the backward pass.
        outputs (np.ndarray): Outputs of the layer before the activation. Needed in the backward pass.
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
        self.__weights = None
        self.__bias = None
        self.__has_weights = has_weights  # Whether the layer has weights and biases.
        self.trainable = trainable
        self.initialized = False
        self.__activation = get_activation_function(activation)
        self.inputs = None
        self.outputs = None  # output of the layer before activation

    # Getters
    # -------------------------------------------------------------------------

    @property
    def activation(self) -> ActivationFunction:
        return self.__activation

    @property
    def has_weights(self) -> bool:
        return self.__has_weights

    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    @property
    def bias(self) -> np.ndarray:
        return self.__bias

    # Setters
    # -------------------------------------------------------------------------

    @activation.setter
    def activation(self, activation: str):
        self.__activation = get_activation_function(activation)

    @weights.setter
    def weights(self, weights: np.ndarray):
        self.set_weights(weights=weights)

    @bias.setter
    def bias(self, bias: np.ndarray):
        self.set_weights(bias=bias)

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

    def get_delta(self, d_out: np.ndarray) -> np.ndarray:
        """
        Args:
            d_out: derivative of the cost function with respect to the output of this layer.
        Returns:
            The delta of the layer (d_C/d_z).
        """
        # check if the layer is initialized
        if not self.initialized:
            raise ValueError("The layer is not initialized")

        if self.activation is None:
            delta = d_out
        else:
            activation_gradient = self.activation.gradient(self.outputs)
            # check if the gradient is a matrix of tensors
            #
            # For example, in the case of Dense layers, we have to check if the gradient is a matrix of tensors
            # with shape=(m, n_neurons) or a tensor of jacobian matrices with shape=(m, n_neurons, n_neurons)
            # This is because of activation functions such
            # as softmax which returns a matrix of jacobian matrices.

            if activation_gradient.shape == self.outputs.shape:
                delta = d_out * activation_gradient
            else:
                d_out = d_out[:, np.newaxis, :]
                delta = np.einsum('ijk,ikl->il', d_out, activation_gradient)

                # The above einsum is equivalent (but faster) to the following code:
                # delta = np.empty_like(self.outputs)
                # batch_size = self.input_shape.shape[0]
                # for i in range(batch_size):  # for each sample
                #     delta[i] = d_out[i] @ activation_gradient[i]

        return delta

    @abstractmethod
    def get_d_inputs(self, delta: np.ndarray) -> np.ndarray:
        """
        Returns the derivative of the cost function with respect to the input of the layer.
        Args:
            delta: derivative of the cost function with respect to the output of the layer.
        Returns:
            derivative of the cost function with respect to the input of the layer.
        """

    @abstractmethod
    def summary(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def update(self, optimizer, delta: np.ndarray):
        """Updates the weights and biases of the layer.

        If the layer has no weights, this method does nothing. If the layer has weights,
        the weights and biases are updated according to the optimizer after computing the
        gradient of the cost function with respect to the weights and biases.

        Args:
            optimizer (Optimizer): optimizer to use.
            delta (np.ndarray): gradient of the cost function with respect to the output of the layer (without
                the activation function).
        """
        raise NotImplementedError

    @abstractmethod
    def count_params(self) -> int:
        """Returns the number of parameters of the layer.

        If the layer has no weights, this method returns 0.
        """

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __call__(self, inputs, *args, **kwargs):
        return self.forward(inputs, *args, **kwargs)
