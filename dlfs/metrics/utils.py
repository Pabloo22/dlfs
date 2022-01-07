from typing import Union

from dlfs.metrics import Metric, Accuracy
from dlfs.losses import *


def get_metric(name: str) -> Union[Metric, LossFunction]:
    """
    Returns a metric object by name.
    """

    if name == 'accuracy':
        return Accuracy()
    elif name == 'mae':
        return MAE()
    elif name == 'mse':
        return MSE()
    else:
        raise ValueError(f'Unknown metric: {name}')
