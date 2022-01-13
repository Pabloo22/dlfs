# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the convolutional layers classes (only the 2D convolution has been implemented by now)."""

import numpy as np
from typing import Tuple, Union
from .layer import Layer
from dlfs.activation_functions import ActivationFunction
from dlfs.convolutioners import Convolutioner, get_convolution


class Conv2D(Layer):
    """A 2D convolution layer.

    Conv2D is analogous to the Dense layer. They differ in the fact that Conv2D takes into account
    the spatial location. It also concerns some other parameters such as stride, kernel size, padding,
    etc; which are essential characteristics in order to carry out a convolution.

    Technical details (assuming data_format='channels_last'):
        - The input is a 4D tensor (batch, height, width, number of input channels).
        - The output is a 4D tensor (batch, height, width, n_filters). (see _get_output_shape for more details)
        - Each kernel is a 3D tensor (kernel_height, kernel_width, number of input channels).
        - The weights are a 4D tensor (kernel_height, kernel_width, number of input channels, n_filters).
        - The bias is a 1D tensor (n_filters,).
        - The layer performs a '3D' convolution on the input tensor for each kernel.



    Args:
        kernel_size (tuple): tuple of 2 integers, specifying the height and width of the 2D convolution window.
        n_filters (int): Number of filters/kernels.
        stride (tuple or int): specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for all spatial dimensions
        padding (bool, tuple or int): If True, add padding to the input so that the output has the same shape as the
            input (assuming stride = 1). If padding is a tuple of two integers, this defines the amount of padding
            to add to the top, bottom, left and right of the input. If padding is an integer, this number of zeros
            is added to the input on both sides.
        activation (ActivationFunction): Activation function
        use_bias (bool): Whether to use bias
        convolution_type (str): convolution mode. Can be 'winograd', 'direct' or 'patch'.
        name (str): Name of the layer
        input_shape (tuple): shape of the input
        weights_init (str): Initialization method for the weights
        bias_init (str): Initialization method for the bias

    Attributes:
        n_filters (int): Number of filters
        kernel_size (tuple): tuple of 2 integers, specifying the height and width of the 2D convolution window.
        stride (tuple): specifying the strides of the convolution along the height and width.
        padding (tuple): tuple of 2 integers, specifying the padding of the convolution along the height and width.
            This values are computed from the stride and kernel size in order to ensure that the output has the same
            shape as the input (if padding = True).
        activation (ActivationFunction): Activation function
        use_bias (bool): Whether to use bias.
        convolution_type (str): convolution mode. Recommended to be 'winograd'.
        name (str): Name of the layer
        blocksize (Tuple[int, int]): the size of the block, only used with Winograd.
        data_format (str): the data format of the input. Can be 'channels_last' or 'channels_first'.

    Raises:
        ValueError: If using padding and stride != 1.
    """

    def __init__(self,
                 n_filters: int,
                 kernel_size: Union[Tuple[int, int], int],
                 stride: Union[Tuple[int, int], int] = (1, 1),
                 padding: Union[bool, tuple, int] = False,
                 activation: str = None,
                 use_bias: bool = True,
                 convolution_type: str = "simple",
                 name: str = "Conv2D",
                 input_shape: tuple = None,
                 weights_init: str = "glorot_uniform",
                 bias_init: str = "zeros",
                 blocksize: Tuple[int, int] = None,
                 data_format: str = "channels_last"):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        if n_filters < 0:
            raise ValueError("The number of filters must be greater than 0")

        if len(kernel_size) != 2:
            raise ValueError("The kernel size should be a tuple of two integers")

        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError("The kernel size should be greater than 0")

        input_shape = None if input_shape is None else (None, *input_shape)
        output_shape = None
        if isinstance(padding, bool):
            padding = (0, 0) if padding is False else (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            padding = (padding, padding) if isinstance(padding, int) else padding

        # get the output shape if the input shape is known
        if input_shape is not None:
            output_height = (input_shape[1] - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
            output_width = (input_shape[2] - kernel_size[1] + 2 * padding[1]) // stride[1] + 1
            output_shape = (None, output_height, output_width, n_filters)

        super().__init__(input_shape=input_shape,
                         output_shape=output_shape,
                         activation=activation,
                         name=name)

        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding
        self.convolution_type = convolution_type
        self.blocksize = blocksize
        self.use_bias = use_bias
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.forward_conv = None
        self.backward_conv = None
        self.update_conv = None
        self.data_format = data_format
        self._batch_count = 0  # needed to preprocess the data in the winograd algorithm

    def initialize(self, input_shape: tuple, weights: np.ndarray = None, bias: np.ndarray = None):
        """Initializes the layer.

        If weights and bias are not provided, they are initialized using the specified initialization method.

        Args:
            input_shape (tuple): input shape of the layer, it has the form (n_samples, height, width, n_channels)
            weights (np.ndarray): weights of the layer (optional, recommended to be None).
            bias (np.ndarray): bias of the layer (optional, recommended to be None).

        Raises:
            ValueError: if the input shape is not valid.
            ValueError: if the weights and bias are not of the shape:
                (input_shape[3], self.kernel_size[0], self.kernel_size[1], self.n_filters),
                (self.n_filters,), respectively.
            ValueError: if trying to set bias and `self.use_bias` is False.
        """

        # check if the input shape is correct
        if len(input_shape) != 4:
            raise ValueError("The input shape should be a tuple of four integers: "
                             "(n_samples, height, width, n_channels)")

        self.input_shape = input_shape
        self.forward_conv = get_convolution(self.convolution_type,
                                            input_shape[1:],
                                            self.kernel_size,
                                            self.padding,
                                            self.stride,
                                            self.blocksize)

        self.output_shape = self._get_output_shape()

        if self.data_format == "channels_first":
            weights_shape = (self.n_filters, input_shape[3], *self.kernel_size)
        else:
            weights_shape = (*self.kernel_size, input_shape[3], self.n_filters)

        # initialize weights
        if weights is not None:
            if weights.shape != weights_shape:
                raise ValueError(f"The shape of the weights should be "
                                 "(n_filters, kernel_height, kernel_width, n_channels_previous_layer). "
                                 f"Got {weights.shape}, expected {weights_shape}")
            self.weights = weights
        elif self.weights_init == "xavier":
            self.weights = np.random.normal(loc=0,
                                            scale=np.sqrt(
                                                1 / (input_shape[3] * self.kernel_size[0] * self.kernel_size[1])),
                                            size=weights_shape)
        elif self.weights_init == "zeros":
            self.weights = np.zeros(weights_shape)
        elif self.weights_init == "ones":
            self.weights = np.ones(weights_shape)
        elif self.weights_init == "uniform":
            self.weights = np.random.uniform(low=-1, high=1, size=weights_shape)
        elif self.weights_init == "normal":
            self.weights = np.random.normal(loc=0, scale=1, size=weights_shape)
        elif self.weights_init == "glorot_uniform":
            self.weights = np.random.uniform(low=-np.sqrt(6 / (input_shape[3] + self.n_filters)),
                                             high=np.sqrt(6 / (input_shape[3] + self.n_filters)),
                                             size=weights_shape)
        else:
            raise ValueError("Unknown weights initialization")

        if self.use_bias:
            bias_shape = (self.n_filters,)
            # initialize bias
            if bias is not None:
                if bias.shape != bias_shape:
                    raise ValueError(f"The shape of the bias should be "
                                     "(n_channels_current_layer). "
                                     f"Got {bias.shape}, expected {bias_shape}")
                self.bias = bias
            elif self.bias_init == "zeros":
                self.bias = np.zeros(bias_shape)
            elif self.bias_init == "ones":
                self.bias = np.ones(bias_shape)
            elif self.bias_init == "uniform":
                self.bias = np.random.uniform(low=-1, high=1, size=bias_shape)
            elif self.bias_init == "normal":
                self.bias = np.random.normal(loc=0, scale=1, size=bias_shape)
            else:
                raise ValueError("Unknown bias initialization")
        else:
            if bias is not None:
                raise ValueError("The bias should be None if the layer is not using bias")

        # INITIALIZE BACKWARD CONVOLUTION
        # compute the padding needed for the backward convolution
        padding_backward = ((self.kernel_size[0] - 1), (self.kernel_size[1] - 1))

        # initialize the backward convolution. This convolution is used to compute the gradient of the loss
        # with respect to the input.
        self.backward_conv = get_convolution(self.convolution_type,
                                             image_size=self.output_shape,  # delta.shape
                                             kernel_size=self.kernel_size,
                                             stride=self.stride,
                                             padding=padding_backward,
                                             data_format=self.data_format)

        # INITIALIZE UPDATE CONVOLUTION
        # This convolution is used to compute the gradient of the loss with respect to the weights.

        # The image size is the size of the output of the convolution but after applying the padding.
        if self.data_format == "channels_first":
            image_height, image_width = (self.output_shape[1], self.output_shape[2])
        else:
            image_height, image_width = (self.output_shape[2], self.output_shape[3])

        # compute the image size after applying the padding
        image_size = (image_height + 2 * padding_backward[0], image_width + 2 * padding_backward[1])

        self.update_conv = get_convolution(self.convolution_type,
                                           image_size=image_size,
                                           kernel_size=self.kernel_size,
                                           stride=self.stride,
                                           data_format=self.data_format)

        self.initialized = True

    def set_weights(self, weights: np.ndarray = None, bias: np.ndarray = None):
        """Sets the weights and bias of the layer.

        Args:
            weights (np.ndarray): weights of the layer.
            bias (np.ndarray): bias of the layer.

        Raises:
            ValueError: if the shape of the weights or bias is not correct.
            ValueError: if trying to set bias and `self.use_bias` is False.
        """
        if weights is not None:
            # check if the weights shape is correct
            if self.data_format == "channels_first":
                weights_shape = (self.n_filters, self.input_shape[1], self.kernel_size[0], self.kernel_size[1])
            else:
                weights_shape = (self.n_filters, self.kernel_size[0], self.kernel_size[1], self.input_shape[-1])
            if weights.shape != weights_shape:
                raise ValueError(f"The shape of the weights should be {weights_shape}, got {weights.shape}.")
            self.weights = weights

        if bias is not None:

            if not self.use_bias:
                raise ValueError("The layer does not use bias.")

            # check if the bias shape is correct
            bias_shape = (self.n_filters,)
            if bias.shape != bias_shape:
                raise ValueError(f"The shape of the bias should be "
                                 "(n_channels_current_layer). "
                                 f"Got {bias.shape}, expected {bias_shape}")

            self.bias = bias

    def _convolve(self,
                  x: np.ndarray,
                  kernel: np.ndarray,
                  convolutioner: Convolutioner,
                  data_format: str = "channels_last") -> np.ndarray:
        """Convolve a batch of multichannel images given a 'Convolutioner'

        Args:
            x (np.ndarray): batch of images.
            kernel (np.ndarray): kernel to convolve with.
            convolutioner (Convolutioner): convolutioner used to convolve the images.

        Returns:
            np.ndarray: batch of convolved images.
        """

        if data_format == "channels_last":
            outputs = np.array([convolutioner.convolve(x,
                                                       kernel[..., i],
                                                       batch_count=self._batch_count,
                                                       data_format=data_format) + self.bias[i]
                                for i in range(kernel.shape[-1])])

            # the shape of self.outputs.shape is (n_filters, batch_size, height, width). So we need to move the axis
            # to (batch_size, height, width, n_filters)
            outputs = np.moveaxis(outputs, 0, -1)
        else:  # data_format == "channels_first"
            outputs = np.array([convolutioner.convolve(x,
                                                       kernel[i],
                                                       batch_count=self._batch_count,
                                                       data_format=data_format) + self.bias[i]
                                for i in range(kernel.shape[0])])

            # the shape of self.outputs.shape is (n_filters, batch_size, height, width). So we need to move the axis
            # to (batch_size, n_filters, height, width)
            outputs = np.moveaxis(outputs, 0, 1)

        return outputs

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass

        Args:
            x: input data
            training: for compatibility with other layers

        Returns:
            output data
        """
        self.inputs = x
        self.outputs = self._convolve(x, self.weights, self.forward_conv, data_format=self.data_format)

        # update the batch count
        self._batch_count += 1
        # Take the modulo 2 to reduce memory usage. We only need to pass a batch count different from the previous one
        # to the update convolution, in order to indicate that the batch has changed. That's why we can do this.
        self._batch_count %= 2

        return self.outputs if self.activation is None else self.activation(self.outputs)

    def get_d_inputs(self, delta: np.ndarray) -> np.ndarray:
        """Returns the derivative of the cost function with respect to the input of the layer.

        Args:
            delta: derivative of the cost function with respect to the output of the layer.
        """
        # 180° rotation of the filters
        rotated_weights = np.rot90(self.weights, 2, axes=(1, 2)) if self.data_format == "channels_last" else \
            np.rot90(self.weights, 2, axes=(2, 3))
        a = 0
        a += 1

        # The current shape of the rotated weights is (height, width, n_input_channels, n_output_channels).
        # We need to exchange the number of input and output channels because we are in the backward pass.
        if self.data_format == "channels_last":
            rotated_weights = np.moveaxis(rotated_weights, 2, -1)
        else:
            rotated_weights = np.moveaxis(rotated_weights, 2, 0)
            rotated_weights = np.moveaxis(rotated_weights, -1, 0)

        d_inputs = self._convolve(delta, rotated_weights, self.backward_conv, data_format=self.data_format)
        return d_inputs

    def count_params(self) -> int:
        return self.weights.size + self.bias.size

    def _get_output_shape(self) -> Tuple[int, int, int, int]:
        """Returns the shape of the output of the layer, taking into account the already defined attributes.

        If the convolution mode is not 'winograd', the output shape could be computed as usual:
        batch_size = self.input_shape[0]
        output_height = (self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        output_width = (self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

        return batch_size, output_height, output_width, self.n_filters

        However, if the convolution mode is 'winograd', the output shape is computed differently (see the
        `get_output_shape` method of the `WinogradConvolutioner` class for more details). For
         this reason we need to call the `get_output_shape` method of the `Convolution` class."""
        batch_size = self.input_shape[0]
        output_height, output_width = self.forward_conv.get_output_shape()
        return batch_size, output_height, output_width, self.n_filters

    def update(self, optimizer, delta: np.ndarray):
        """Updates the weights and biases of this layer.

        See `base class` for more details.

        Args:
            optimizer (Optimizer): optimizer used to update the weights and biases
            delta: delta of the loss with respect to the output of this layer
        """
        # check if the layer is initialized
        if not self.initialized:
            raise ValueError("The layer is not initialized")

        # Compute the gradient of the loss with respect to the weights and biases

        if self.data_format == "channels_last":
            # The self.inputs.shape is (batch_size, inputs_height, inputs_width, n_channels)
            # The delta.shape is (batch_size, inputs_height, inputs_width, n_filters)
            # The self.weights.shape is (kernel_height, kernel_width, n_channels, n_filters)

            # The gradient of the loss with respect to the weights is computed as:
            # d_w = convolve(inputs, delta)

            # We can not perform this operation directly because the convolution function expects the inputs to
            # have the shape (batch_size, inputs_height, inputs_width, n_channels) and the kernel (deltas) to have
            # the shape (kernel_height, kernel_width, n_channels). Thus, we have to use a list comprehension to
            # perform 'n_filters*batch_size' convolutioners and then sum the results and move the axes to the correct
            # positions.

            dw = np.zeros(self.weights.shape)
            for i, image in enumerate(self.inputs):
                for j in range(image.shape[2]):  # image.shape[2] = number of channels
                    for k in range(self.n_filters):
                        dw[:, :, j, k] += self.update_conv(image[:, :, j], delta[i, :, :, k], using_batches=False)

            dw = dw / self.inputs.shape[0]
            db = np.sum(delta, axis=(0, 1, 2))

        optimizer.update(self, (dw, db))

    def summary(self):
        return f"Conv2D: {self.name}, Output shape: {self.output_shape}"
