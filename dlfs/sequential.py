import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from tqdm import tqdm

from dlfs.layers import Layer
from dlfs.optimizers import Optimizer
from dlfs.losses.loss_function import LossFunction
from dlfs.metrics import Metric, get_metric


class Sequential:
    """
    A sequential model is a linear stack of layers.
    """

    layers: List[Layer]
    name: str
    loss: LossFunction or None
    optimizer: Optimizer or None
    metrics: Dict[str, Metric] or None
    trainable: bool

    def __init__(self, layers: List[Layer] = None, name: str = "Sequential"):

        # check that the first layers has set the input shape
        if layers and layers[0].input_shape is None:
            raise ValueError("The first layer must have an input shape")

        self.layers = []
        if layers is not None:
            for layer in layers:
                self.add(layer)
        self.name = name
        self.loss = None
        self.optimizer = None
        self.metrics = None
        self.trainable = True

    def add(self, layer: Layer):
        """
        Add a layer to the model
        Args:
            layer (Layer): the layer to add
        """

        # check that the first layers has set the input shape
        if not self.layers and layer.input_shape is None:
            raise ValueError("The first layer must have an input shape")

        # initialize the layer
        if layer.input_shape is None:
            layer.initialize(input_shape=self.layers[-1].output_shape)
        else:
            layer.initialize(input_shape=layer.input_shape)

        self.layers.append(layer)

    def compile(self, optimizer: Optimizer, loss: LossFunction, metrics: List[str] = None):
        """
        Compile the model

        Args:
            optimizer: the optimizer to use
            loss: the loss function to use
            metrics: the metrics to use. The metrics by default are the ones used by the loss function
            Allowed values are:
                - "accuracy"
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = {} if metrics is None else {metric: get_metric(metric) for metric in metrics}

    def summary(self):
        """
        Print the summary of the model as a table
        """
        print(f"Model: {self.name}")
        print("-" * len(self.name))
        print("Layers:")
        for layer in self.layers:
            print("-" * len(layer.summary()))
            print(f"{layer.summary()}")

        print("-" * len(layer.summary()))
        print("Total params:", sum(layer.count_params() for layer in self.layers))
        print("Trainable params:", sum(layer.count_params() for layer in self.layers if layer.trainable))
        print("-" * len(layer.summary()))

    def get_gradients(self, y_pred: np.ndarray, y_true: np.ndarray) -> List[np.ndarray]:
        """
        Backpropagation of the loss function
        Args:
            y_pred: the predictions
            y_true: the true labels

        Returns:
            the gradients of the loss function
        """
        # backward pass
        # initialize the gradients
        gradients = [np.ndarray([])] * len(self.layers)
        gradients[-1] = self.loss.compute_loss(y_pred, y_true)

        # compute the gradients
        for i in range(len(self.layers) - 1, 0, -1):
            # store in gradients[i -1] the gradient of the loss function with respect to the output of the layer i
            # gradients[i - 1] = (with respect to weights, with respect to biases)
            gradients[i - 1] = self.layers[i].backward(gradients[i])

        return gradients

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 1,
            batch_size: int = 32,
            verbose: int = 1,
            validation_data: Tuple[np.ndarray, np.ndarray] = None,
            validation_split: float = 0.0,
            shuffle: bool = True,
            initial_epoch: int = 0) -> Dict[str, List[float]]:
        """
        Fit the model to the data

        Args:
            x: the input data. With shape (n_samples, n_features)
            y: the labels
            epochs: the number of epochs to train the model
            batch_size: the batch size
            verbose: the verbosity mode. 0 is progress bar, 1 is one line per epoch, 2 is one line per batch
            validation_data: the validation data
            validation_split: the validation split
            shuffle: whether to shuffle the data
            initial_epoch: the initial epoch
        """
        # CHECKS:
        # check if the model has been compiled
        if self.optimizer is None or self.loss is None:
            raise ValueError("You must compile the model before training")

        using_validation_data = False
        # if validation data is not provided, split the data into train and validation
        if validation_split > 0:
            if validation_data is not None:
                raise ValueError("validation_data and validation_split cannot be used together")
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split)
            using_validation_data = True

        if validation_data is not None:
            x_val, y_val = validation_data
            using_validation_data = True

        if using_validation_data and verbose == 0:
            raise ValueError("validation_data and verbose=0 cannot be used together")

        # INITIALIZATION:
        # --------------------------------------------------

        # Update the input_shape of the layers to take into account the batch_size
        for layer in self.layers:
            layer.input_shape = (batch_size, *layer.input_shape[1:])

        # initialize the history
        history = {'loss': [], 'val_loss': []}
        for metric in self.metrics:
            history[metric] = []
            history['val_' + metric] = []

        # set the optimizers
        for layer in self.layers:
            layer.optimizer = self.optimizer

        # TRAINING:
        # --------------------------------------------------
        # create the progress bar if verbose is 0
        r = range(initial_epoch, epochs)
        range_epochs = r if verbose > 0 else tqdm(r)

        # loop over the number of epochs
        for epoch in range_epochs:
            # initialize the total loss for the epoch if verbose is 2
            if verbose > 0:
                epoch_loss = 0.0
                epoch_metrics = {metric: 0.0 for metric in self.metrics}

            # loop over the data in batches
            total_data_used_per_epoch = 0
            for x_batch, y_batch in self.batch_generator(x, y, batch_size, shuffle):

                total_data_used_per_epoch += x_batch.shape[0]

                # forward pass
                y_pred = self.predict(x_batch, training=True)

                # backward pass: get the gradients
                gradients = self.get_gradients(y_pred, y_batch)

                # update the parameters
                for i, layer in enumerate(self.layers):
                    if layer.trainable:
                        layer.update(gradients[i])

                # compute the loss for the batch
                loss = self.loss.compute_loss(y_pred, y_batch)
                if using_validation_data:
                    val_loss = self.loss.compute_loss(self.predict(x_val, training=False), y_val)

                # update the total loss (if verbose is 2)
                if verbose > 0:
                    epoch_loss += loss
                    for metric in self.metrics:
                        epoch_metrics[metric] += self.metrics[metric].compute_metric(self, x_batch, y_batch)

                # print the loss and the metrics per batch (if verbose is 1)
                if verbose == 1:
                    if using_validation_data:
                        print(f"Batch ({total_data_used_per_epoch}/{x.shape[0]}) - loss: {loss:.4f} - val_loss: {val_loss:.4f}")
                    else:
                        print(f"Batch ({total_data_used_per_epoch}/{x.shape[0]}) - loss: {loss:.4f}")
                    metrics_list = [f'{metric}: {self.metrics[metric].compute_metric(y_pred, y_batch):.4f}'
                                    for metric in self.metrics]
                    print(f"\t{', '.join(metrics_list)}")

            # print the metrics (if verbose is 1 or 2)
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"\tTrain loss: {epoch_loss}")
                for metric in self.metrics:
                    print(f"\t{metric}: {epoch_metrics[metric] / len(x)}")

                if using_validation_data:
                    y_pred_val = self.predict(x_val)
                    val_loss = self.loss.compute_loss(y_pred_val, y_val)
                    print(f"\tValidation loss: {val_loss}")
                    # add the validation loss to the history
                    history['val_loss'].append(epoch_loss)

                    for metric in self.metrics:
                        val_metric = self.metrics[metric].compute_metric(self, x_val, y_val)
                        print(f"\t{metric}: {val_metric:.4f}")
                        # add the validation metric to the history
                        history['val_' + metric].append(val_metric)

                    # append the val_loss and val_metrics to the history
                    history['val_loss'].append(val_loss)
                    for metric in self.metrics:
                        history['val_' + metric].append(self.metrics[metric].compute_metric(y_pred_val, y_val))
            print("")

            # save the history
            history['loss'].append(epoch_loss)
            for metric in self.metrics:
                history[metric].append(epoch_metrics[metric] / len(x))

        return history

    @staticmethod
    def batch_generator(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
        """
        Generates batches of data
        Args:
            x: the input data
            y: the labels
            batch_size: the batch size
            shuffle: whether to shuffle the data

        Yields:
            x_batch: the batch of input data
            y_batch: the batch of labels
        """
        # get the number of batches
        n_batches = len(x) // batch_size

        if shuffle:
            x = np.random.permutation(x)
            y = np.random.permutation(y)

        # loop over the batches
        for i in range(0, n_batches * batch_size, batch_size):
            # get the batch data
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            # yield the batch
            yield x_batch, y_batch

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, verbose: int = 1) -> float:
        """
        Args:
            x: the input data
            y: the labels
            batch_size: the batch size
            verbose: the verbosity mode (0 or 1)
        """
        # initialize the total loss and the number of batches
        total_loss = 0.0
        n_batches = len(x) // batch_size

        # loop over the batches
        for i in range(0, n_batches * batch_size, batch_size):
            # get the batch data
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            # compute the loss for the batch
            loss = self.loss.compute_loss(self, x_batch, y_batch)
            # update the total loss
            total_loss += loss
        # compute the average loss
        total_loss /= len(x)
        # print the metrics
        if verbose:
            # TODO: print the metrics
            pass
        return total_loss

    def predict(self, x: np.ndarray, batch_size: int = 32, verbose: int = 1, training: bool = False) -> np.ndarray:
        """

        Args:
            x: the input data
            batch_size: the batch size
            verbose: the verbosity mode (0 or 1)
        """

        # initialize the predictions
        y_pred = []

        # loop over the batches
        for i in range(0, len(x), batch_size):
            # get the batch data
            x_batch = x[i:i + batch_size]
            # get the predictions for the batch
            y_batch_pred = self.predict_batch(x_batch)
            # update the predictions
            y_pred.append(y_batch_pred)
        # concatenate the predictions
        y_pred = np.concatenate(y_pred)
        # print the metrics
        if verbose:
            # TODO: print the metrics
            pass

        return y_pred

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        """

        Args:
            x : the input data
        Returns:
            y_pred : the predictions
        """

        last_input = x
        for layer in self.layers:
            last_input = layer.forward(last_input)
        return last_input

    def save(self, path: str):
        """
        Args:
            path: the path to save the model
        """

#
# if __name__ == '__main__':
#     length = 10
#     array = np.zeros(length)
#     print(array)
#     array[-1] = 1
#
#     for i in range(length - 2, -1, -1):
#         array[i] = array[i + 1] + 1
#     print(array)
