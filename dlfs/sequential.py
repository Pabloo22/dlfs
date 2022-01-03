from collections import deque
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle


from dlfs.layers import Layer
from dlfs.optimizers import Optimizer, get_optimizer
from dlfs.losses import LossFunction, get_loss_function
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
        if not layer.initialized:
            if layer.input_shape is None:
                layer.initialize(input_shape=self.layers[-1].output_shape)
            else:
                layer.initialize(input_shape=layer.input_shape)

        self.layers.append(layer)

    def compile(self, optimizer: Optimizer or str, loss: LossFunction or str, metrics: List[str] = None):
        """
        Compile the model

        Args:
            optimizer: the optimizer to use
            loss: the loss function to use
            metrics: the metrics to use. The metrics by default are the ones used by the loss function
            Allowed values are:
                - "accuracy"
        """
        self.optimizer = get_optimizer(optimizer) if isinstance(optimizer, str) else optimizer
        self.loss = get_loss_function(loss) if isinstance(loss, str) else loss
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

        print("-" * len(self.name))
        print("Total params:", sum(layer.count_params() for layer in self.layers))
        print("Trainable params:", sum(layer.count_params() for layer in self.layers if layer.trainable))
        print("-" * len(self.name))

    def get_deltas(self, y_pred: np.ndarray, y_true: np.ndarray) -> deque:
        """
        Backpropagation of the loss function
        Args:
            y_pred: the predictions
            y_true: the true labels

        Returns:
            the deltas of the loss function (as a deque)
        """

        # initialize the deltas
        deltas = deque()

        last_layer = self.layers[-1]
        dz_da = last_layer.get_dz_da()

        # initialize the deltas of the last layer
        if last_layer.activation is None:
            deltas.appendleft(self.loss.gradient(y_pred, y_true))
        else:
            deltas.appendleft(self.loss.gradient(y_pred, y_true) * last_layer.activation.derivative(last_layer.z))

        # backward pass
        for layer in reversed(self.layers[:-1]):
            delta = layer.get_delta(deltas[0], dz_da)
            deltas.appendleft(delta)
            dz_da = layer.get_dz_da()

        return deltas

    def train_on_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Dict[str, float]:
        """
        Train the model on a batch of data

        Args:
            x_batch: the batch of inputs
            y_batch: the batch of expected outputs

        Returns:
            the loss and the metrics
        """

        # forward pass
        y_pred = self.predict(x_batch, training=True)

        # backward pass: get the deltas
        deltas: deque = self.get_deltas(y_pred, y_batch)

        # backward pass: update the weights
        for i, layer in enumerate(self.layers):
            if layer.trainable:
                layer.update(deltas[i])

        # compute the loss for the batch
        loss = self.loss.compute_loss(y_pred, y_batch)

        # compute the metrics for the batch
        metrics = {metric.name: metric.compute_metric(y_pred, y_batch) for metric in self.metrics.values()}

        return {**{'loss': loss}, **metrics}

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 1,
            batch_size: int = 32,
            verbose: int = 1,
            validation_data: Tuple[np.ndarray, np.ndarray] = None,
            validation_split: float = 0.,
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
        np.set_printoptions(precision=4)

        # Update the input_shape of the layers to take into account the batch_size
        for layer in self.layers:
            layer.input_shape = (batch_size, *layer.input_shape[1:])

        # Update the output_shape of the layers to take into account the batch_size
        for layer in self.layers:
            layer.output_shape = (batch_size, *layer.output_shape[1:])

        # initialize the history
        history = {"loss": [], "val_loss": []}
        for metric in self.metrics:
            history[metric] = []
            history["val_" + metric] = []

        # set the optimizers
        for layer in self.layers:
            layer.optimizer = deepcopy(self.optimizer)

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

                metrics = self.train_on_batch(x_batch, y_batch)
                loss = metrics["loss"]
                if using_validation_data:
                    val_loss = self.loss.compute_loss(self.predict(x_val, training=False), y_val)

                # update the total loss for the epoch if verbose is not zero
                if verbose > 0:
                    epoch_loss += loss
                    for metric in self.metrics:
                        epoch_metrics[metric] += metrics[metric]

                # print the loss and the metrics per batch (if verbose is 1)
                if verbose == 1:
                    if using_validation_data:
                        print(f"Batch ({total_data_used_per_epoch}/{x.shape[0]}) "
                              f"- loss: {loss} - val_loss: {val_loss}")
                    else:
                        print(f"Batch ({total_data_used_per_epoch}/{x.shape[0]}) - loss: {loss}")
                    print(f"\t{', '.join([f'{metric}: {metrics[metric]:.4f}' for metric in self.metrics])}")

            # print the metrics (if verbose is 1 or 2)
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"\tTrain loss: {epoch_loss / (len(x) // batch_size):.4f}")
                for metric in self.metrics:
                    print(f"\t{metric}: {epoch_metrics[metric] / len(x)}")

                if using_validation_data:
                    y_pred_val = self.predict(x_val)
                    val_loss = self.loss.compute_loss(y_pred_val, y_val)
                    print(f"\tValidation loss: {val_loss}")
                    # add the validation loss to the history
                    history["val_loss"].append(epoch_loss)

                    for metric in self.metrics:
                        val_metric = self.metrics[metric].compute_metric(self, x_val, y_val)
                        print(f"\t{metric}: {val_metric:.4f}")
                        # add the validation metric to the history
                        history["val_" + metric].append(val_metric)

                    # append the val_loss and val_metrics to the history
                    history["val_loss"].append(val_loss)
                    for metric in self.metrics:
                        history["val_" + metric].append(self.metrics[metric].compute_metric(y_pred_val, y_val))
            print("")

            # save the history
            history["loss"].append(epoch_loss)
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
            # shuffle the data
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]

        # loop over the batches
        for i in range(0, n_batches * batch_size, batch_size):
            # get the batch data
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            # yield the batch
            yield x_batch, y_batch

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = None, verbose: int = 1) -> dict:
        """
        Args:
            x: the input data
            y: the labels
            batch_size: the batch size (default: len(x))
            verbose: the verbosity mode (0 or 1)

        Returns:
            The average loss and the average metrics of the model on the given data
        """
        # get the batch size
        batch_size = batch_size or len(x)

        # initialize the metrics
        avg_loss = 0.0
        avg_metrics = {metric: 0.0 for metric in self.metrics}

        n_batches = len(x) // batch_size

        # loop over the batches
        for i in range(0, n_batches * batch_size, batch_size):
            # get the batch data
            y_pred = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # forward pass
            y_pred = self.predict(y_pred)

            # compute the loss for the batch
            loss = self.loss.compute_loss(y_pred, y_batch)

            # compute the metrics for the batch
            metrics = {metric: self.metrics[metric].compute_metric(y_pred, y_batch) for metric in self.metrics}

            # update the loss and the metrics
            avg_loss += 1/(i + 1) * (loss - avg_loss)
            for metric in self.metrics:
                avg_metrics[metric] += 1/(i + 1) * (metrics[metric] - avg_metrics[metric])

            # print the loss and the metrics per batch (if verbose is 1)
            if verbose > 0 and batch_size != len(x):
                print(f"Batch ({i // batch_size + 1}/{n_batches}) - loss: {loss}")
                print(f"\t{', '.join([f'{metric}: {metrics[metric]:.4f}' for metric in self.metrics])}")

        # print final loss and metrics
        if verbose > 0:
            print(f"Average loss: {avg_loss}")
            print(f"{', '.join([f'{metric}: {avg_metrics[metric]:.4f}' for metric in self.metrics])}")

        return {'loss': avg_loss, **avg_metrics}

    def predict(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """

        Args:
            x : the input data
            training: whether the model is in training mode
        Returns:
            y_pred : the predictions
        """

        last_input = x
        for layer in self.layers:
            last_input = layer.forward(last_input, training)
        return last_input

    def set_weights(self, weights: list, layers: list = None, only_weights: bool = False, only_biases: bool = False):
        """
        Sets the weights and biases of the model. The weights and biases are in the same order as the layers.
        Args:
            weights: the weights of the model (list of numpy arrays) [layer_0_weights, layer_0_bias, ...]
                The weights and biases of all the layers that have them must be included in the list
            layers: the layers of the model (list of Layer objects) which you want to set the weights of
                (default: all layers with weights)
            only_weights: whether to set only the weights. If True, the weights list must contain
                only the weights of the layers in the layers list. (default: False)
            only_biases: whether to set only the biases. If True, the weights list must contain
                only the biases of the layers in the layers list. (default: False)
        """
        layers_with_weights = layers or [layer for layer in self.layers if layer.has_weights]

        if only_weights:
            for i, w in enumerate(weights):
                layers_with_weights[i].set_weights(w)
        elif only_biases:
            for i, b in enumerate(weights):
                layers_with_weights[i].set_weights(bias=b)
        else:
            for i, layer in enumerate(layers_with_weights):
                layer.set_weights(weights[i], weights[i + len(layers_with_weights)])

    def save(self, path: str):
        """
        Saves the model to the given path using the pickle module
        Args:
            path: the path to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        """
        Loads the model from the given path using the pickle module
        Args:
            path: the path to load the model
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join([str(layer) for layer in self.layers])})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index: int):
        return self.layers[index]

    def __iter__(self):
        return iter(self.layers)

    def __add__(self, other: 'Sequential'):
        return Sequential(self.layers + other.layers)

    def __iadd__(self, other: 'Sequential'):
        self.layers += other.layers
        return self

    def __copy__(self):
        return deepcopy(Sequential(self.layers))