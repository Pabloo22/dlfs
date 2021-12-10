from . import Metric, Accuracy


def get_metric(name: str) -> Metric:
    """
    Returns a metric object by name.
    """

    if name == 'accuracy':
        return Accuracy()
    else:
        raise ValueError(f'Unknown metric: {name}')
