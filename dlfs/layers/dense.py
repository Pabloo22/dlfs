import numpy as np

from .layer import Layer


class Dense(Layer):
    """
    Dense layer class

    Args:
        n_neurons (int): number of neurons
        activation (str): activation function
        name (str): layer name
    """

    def __init__(self, n_neurons: int, activation: str = None, name: str = "Dense", input_shape: tuple = None,
                 weights_init: str = "xavier", bias_init: str = "zeros"):

        if n_neurons <= 0:
            raise ValueError("The number of neurons should be greater than 0")

        # In order to give an easier way to define the layer, we allow the user to define the input shape
        # without having to specify the batch size, that is why we set the batch size to None.
        input_shape = None if input_shape is None else (None, *input_shape)
        super(Dense, self).__init__(input_shape=input_shape,
                                    output_shape=(None, n_neurons),
                                    activation=activation,
                                    name=name)
        self.__n_neurons = n_neurons
        self.weights_init = weights_init  # recommended: xavier
        self.bias_init = bias_init  # recommended: zeros
        self.weights = None
        self.bias = np.zeros((1, n_neurons))
        self.inputs = None
        self.z = None  # the output of the layer before the activation function

    # Getters
    # ----------------------------------------------------------------------------------------------------

    @property
    def n_neurons(self):
        return self.__n_neurons

    # Methods
    # ----------------------------------------------------------------------------------------------------
    def initialize(self, input_shape: tuple, weights: np.ndarray = None, bias: np.ndarray = None):
        """
        Initialize the layer. Should be called after the input shape is set.

        Args:
            input_shape (tuple): input shape of the layer, it has the form (n_samples (None), n_features)
            weights (np.ndarray): weights of the layer (optional, recommended to be None).
                The weights has the shape (n_neurons_prev_layer, n_neurons_current_layer). Each column of the weights
                matrix represents a neuron and the values of the column are its weights.
            bias (np.ndarray): bias of the layer (optional, recommended to be None). The bias has the shape
                (1, n_neurons_current_layer).
        """
        # check if the input shape is correct
        if len(input_shape) != 2:
            raise ValueError("The input shape is incorrect")

        self.input_shape = input_shape

        if weights is not None:
            self.weights = weights
        elif self.weights_init == "xavier":
            # The recommended initialization [https://www.deeplearning.ai/ai-notes/initialization/]
            self.weights = np.random.normal(loc=0,
                                            scale=np.sqrt(2 / (input_shape[1] + self.n_neurons)),
                                            size=(input_shape[1], self.n_neurons))
        elif self.weights_init == "zeros":
            self.weights = np.zeros((input_shape[1], self.n_neurons))
        elif self.weights_init == "ones":
            self.weights = np.ones((input_shape[1], self.n_neurons))
        elif self.weights_init == "normal":
            self.weights = np.random.normal(loc=0,
                                            scale=1,
                                            size=(input_shape[1], self.n_neurons))
        elif self.weights_init == "uniform":
            self.weights = np.random.uniform(low=-1,
                                             high=1,
                                             size=(input_shape[1], self.n_neurons))
        else:
            raise ValueError("The weights initializer is incorrect")

        if bias is not None:
            self.bias = bias
        elif self.bias_init == "zeros":
            # the recommended initialization [https://www.deeplearning.ai/ai-notes/initialization/]
            self.bias = np.zeros((1, self.n_neurons))
        elif self.bias_init == "ones":
            self.bias = np.ones((1, self.n_neurons))
        elif self.bias_init == "normal":
            self.bias = np.random.normal(loc=0,
                                         scale=1,
                                         size=(1, self.n_neurons))
        elif self.bias_init == "uniform":
            self.bias = np.random.uniform(low=-1,
                                          high=1,
                                          size=(1, self.n_neurons))
        else:
            raise ValueError("The bias initializer is incorrect")

        self.initialized = True

    def forward(self, inputs, training: bool = False) -> np.ndarray:
        """
        Forward pass of the layer.

        Args:
            inputs (np.ndarray): inputs of the layer
            training (bool): for compatibility with other layers

        Returns:
            np.ndarray: outputs of the layer
        """
        # check if the layer is initialized
        if not self.initialized:
            raise ValueError("The layer is not initialized")
        # check if the input shape is correct
        if inputs.shape[1:] != self.input_shape[1:]:
            raise ValueError("The input shape is incorrect")
        # save the inputs
        self.inputs = inputs
        self.z = inputs @ self.weights + self.bias
        return self.z if self.activation is None else self.activation(self.z)

    def get_delta(self, last_delta: np.ndarray, dz_da: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer.
        Args:
            last_delta: gradients of the layer.
            dz_da: next layer in the network. If this is the layer i, the received layer will be the i+1 layer.
                        layers = [layer1, layer2, layer3, ..., layer_i, layer_i+1, ..., layerL]
        Returns:
            The corresponding delta of the layer (d_cost/d_z).
        """

        # check if the layer is initialized
        if not self.initialized:
            raise ValueError("The layer is not initialized")

        # compute the delta

        delta = last_delta * dz_da if last_delta.shape[1:] == self.output_shape[1:] else last_delta @ dz_da

        if self.activation is not None:
            delta *= self.activation.derivative(self.z)

        return delta

    def get_dz_da(self) -> np.ndarray:
        """
        Returns:
            The derivative of the output of this layer (i) with respect to the z of the next layer (i+1).
        """
        # check if the layer is initialized
        if not self.initialized:
            raise ValueError("The layer is not initialized")

        return self.weights.T

    def update(self, delta: np.ndarray):
        """
        Update the weights and biases of the layer.

        Args:
            delta (np.ndarray): delta of the current layer
        """

        # check if the layer is initialized
        if not self.initialized:
            raise ValueError("The layer is not initialized")

        d_weights = (self.inputs.T @ delta) / self.inputs.shape[0]
        d_bias = delta.sum(axis=0, keepdims=True) / self.inputs.shape[0]

        self.optimizer.update((self.weights, self.bias), (d_weights, d_bias))

    def count_params(self) -> int:
        """
        Count the number of parameters of the layer.

        Returns:
            int: number of parameters of the layer
        """
        return self.weights.size + self.bias.size

    def summary(self) -> str:
        """
        Summary of the layer.

        Returns:
            str: summary of the layer
        """
        return f"Dense: {self.__n_neurons} neurons\t output_shape={self.output_shape}\t " \
               f"n_params={self.weights.size + self.bias.size}"
