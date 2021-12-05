from adam import Adam
from optimizer import Optimizer


def get_optimizer(optimizer_name: str, learning_rate: float, **kwargs) -> Optimizer:
    """
    Get an optimizer by name.

    Args:
        optimizer_name (str): The name of the optimizer.
        learning_rate (float): The learning rate.
        **kwargs: Additional arguments for the optimizer.

    Returns:
        The optimizer.
    """
    if optimizer_name == 'adam':
        return Adam(learning_rate, **kwargs)
    else:
        raise ValueError('Invalid optimizer name')
