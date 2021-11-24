from base import ActivationFunction
from relu import ReLU
from softmax import Softmax


def get_activation_function(activation_function_name: str) -> ActivationFunction:
    """
    Returns an activation function object based on the name of the activation function.
    Args:
        activation_function_name: The name of the activation function. The possible names are:
            - "relu": ReLU
            - "softmax": Softmax
    Returns:
        An activation function object.
    """

    if activation_function_name == 'relu':
        return ReLU()
    elif activation_function_name == 'softmax':
        return Softmax()
    else:
        raise ValueError('Activation function name not recognized')
