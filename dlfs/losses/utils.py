from . import MAE, MSE, BinaryCrossEntropy, CategoricalCrossentropy


def get_loss_function(loss_name: str):
    if loss_name == 'mae':
        return MAE()
    elif loss_name == 'mse':
        return MSE()
    elif loss_name == 'binary_crossentropy':
        return BinaryCrossEntropy()
    elif loss_name == 'categorical_crossentropy':
        return CategoricalCrossentropy()
    else:
        raise ValueError(f'Unknown loss function: {loss_name}')
