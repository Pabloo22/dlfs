from . import Optimizer, SGD, SGDMomentum, Adam, Adagrad


def get_optimizer(optimizer_name: str) -> Optimizer:
    """
    Get an optimizer by name.

    Args:
        optimizer_name (str): The name of the optimizer. Valid names are:
            - sgd
            - sgd_momentum
            - adam
            - adagrad

    Returns:
        The optimizer.
    """
    if optimizer_name == "sgd":
        return SGD()
    elif optimizer_name == "sgd_momentum":
        return SGDMomentum()
    elif optimizer_name == "adam":
        return Adam()
    elif optimizer_name == "adagrad":
        return Adagrad()
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")
