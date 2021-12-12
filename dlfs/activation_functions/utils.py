from typing import Union

from . import ActivationFunction, Sigmoid, ReLU, Softmax, Linear, Tanh


def get_activation_function(activation_function_name: Union[str, None]) -> Union[ActivationFunction, None]:
    """
    Returns an activation function object based on the name of the activation function.
    Args:
        activation_function_name: The name of the activation function. The possible names are:
            - "relu": ReLU
            - "softmax": Softmax
            - "sigmoid": Sigmoid
            - "tanh": Tanh
            - "linear": Linear
            - None: No activation function (i.e. the input is passed through, the same as Linear)
    Returns:
        An activation function object or None if there is no activation function (the same as using 'Linear').
    """
    if activation_function_name is None:
        return None

    if activation_function_name == 'relu':
        return ReLU()
    elif activation_function_name == 'softmax':
        return Softmax()
    elif activation_function_name == 'sigmoid':
        return Sigmoid()
    elif activation_function_name == 'linear':
        return Linear()
    elif activation_function_name == 'tanh':
        return Tanh()
    else:
        raise ValueError(f'Unknown activation function name: {activation_function_name}')
