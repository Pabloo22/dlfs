from . import MAE, MSE, BinaryCrossEntropy, CategoricalCrossEntropy


def get_loss_function(loss_name: str):
    if loss_name == 'mae':
        return MAE()
    elif loss_name == 'mse':
        return MSE()
    elif loss_name == 'binary_cross_entropy':
        return BinaryCrossEntropy()
    elif loss_name == 'categorical_cross_entropy':
        return CategoricalCrossEntropy()
    else:
        raise ValueError(f'Unknown loss function: {loss_name}')
