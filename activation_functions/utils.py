from base import ActivationFunction
from relu import ReLU
from softmax import Softmax


def get_activation_function(activation_function_name: str) -> ActivationFunction:

    if activation_function_name == 'relu':
        return ReLU()
    elif activation_function_name == 'softmax':
        return Softmax()
    else:
        raise ValueError('Activation function name not recognized')
