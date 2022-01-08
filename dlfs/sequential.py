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
        self.__trainable = True
        self.counter = 1

    @property
    def trainable(self):
        return self.__trainable

    @trainable.setter
    def trainable(self, value: bool):
        self.__trainable = value
        for layer in self.layers:
            layer.trainable = value

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

        # each layer must have an unique name
        layer.name = f"{layer.name}{self.counter}"
        self.counter += 1

        self.layers.append(layer)

    def concatenate(self, model: 'Sequential'):
        """
        Concatenate two models
        Args:
            model (Sequential): the model to concatenate
        """
        if self.layers[-1].output_shape != model.layers[0].input_shape:
            raise ValueError("The two models must have the same input shape")
        self.layers += model.layers

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

        # check that each layer has an unique name
        names = set()
        for layer in self.layers:
            if layer.name in names:
                raise ValueError(f"The layer name {layer.name} is not unique")
            names.add(layer.name)
            self.optimizer.add_slot(layer)

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
            deltas.appendleft(self.loss.gradient(y_true, y_pred))
        else:
            activation_gradient = last_layer.activation.gradient(last_layer.z)
            # check if the gradient is a matrix of tensors with shape=(m, n_neurons) or a tensor of jacobian matrices
            # with shape (m, n_neurons, n_neurons). (m is the number of samples)
            if activation_gradient.ndim == 2:
                deltas.appendleft(self.loss.gradient(y_true, y_pred) * activation_gradient)
            else:

                loss_gradient = self.loss.gradient(y_true, y_pred)[:, np.newaxis, :]

                delta = np.einsum('ijk,ikl->il', loss_gradient, activation_gradient)

                # The above einsum is equivalent to:
                # delta = np.empty_like(last_layer.z)
                # batch_size = y_true.shape[0]
                # for i in range(batch_size):  # for each sample
                #     delta[i] = loss_gradient[i] @ activation_gradient[i]

                deltas.appendleft(delta)

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
                layer.update(self.optimizer, deltas[i])

        # compute the loss for the batch
        loss = self.loss.compute_loss(y_batch, y_pred)

        # compute the metrics for the batch
        metrics = {metric.name: metric(y_batch, y_pred) for metric in self.metrics.values()}

        return {**{'loss': loss}, **metrics}

    def __check_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if x.shape[1:] != self.layers[0].input_shape[1:]:
            raise ValueError(f"The input shape of the model is {self.layers[0].input_shape}")

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # check if the labels are valid
        if y.shape[1:] != self.layers[-1].output_shape[1:]:
            raise ValueError(f"The output shape of the model is {self.layers[-1].output_shape}")

        return x, y

    def __print_results(self,
                        metrics: Dict[str, float],
                        val_metrics: Dict[str, float] = None,
                        title: str = "Epoch",
                        progress: int = 0,
                        total: int = 0,
                        history: Dict[str, list] = None) -> None:
        """
        Print the results of the training and update the history
        Args:
            metrics: the metrics (and loss) of the current epoch
            val_metrics: the metrics of the validation set
            title: the title of the printed results (Batch or Epoch)
            progress: the progress of the training. The number of batches processed or the number of epochs
            total: the total number of batches or epochs
            history: the history of the training to update
        """

        using_validation = val_metrics is not None
        if using_validation:
            # print the results
            print(f"{title} ({progress}/{total}) - loss: {metrics['loss']} - val_loss: {val_metrics['val_loss']}")
            for metric in self.metrics:
                print(f"- {metric}: {metrics[metric]} - val_{metric}: {val_metrics['val_' + metric]}")
        else:
            # print the results
            print(f"{title} ({progress}/{total}) - loss: {metrics['loss']}")
            for metric in self.metrics:
                print(f"- {metric}: {metrics[metric]}")

        if history is not None:
            for metric in metrics:
                history[metric].append(metrics[metric])
            if using_validation:
                for metric in val_metrics:
                    history[metric].append(val_metrics[metric])
        print("")

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
            verbose: the verbosity mode:
                * 0 doesn't show anything.
                * 1 prints one line per batch.
                * 2 prints one line per epoch.
                * 3 shows the progress bar (per epoch).
            validation_data: the validation data
            validation_split: the validation split
            shuffle: whether to shuffle the data
            initial_epoch: the initial epoch
        """
        # CHECKS:
        # check if the model has been compiled
        if self.optimizer is None or self.loss is None:
            raise ValueError("You must compile the model before training")

        # check if the data dimensions are valid
        x, y = self.__check_data(x, y)

        using_validation_data = True
        # if validation data is not provided, split the data into train and validation
        if validation_split > 0:
            if validation_data is not None:
                raise ValueError("validation_data and validation_split cannot be used together")
            x, x_val, y, y_val = train_test_split(x, y, test_size=validation_split)
        elif validation_data is not None:
            x_val, y_val = validation_data

            # check if the data dimensions are valid
            x_val, y_val = self.__check_data(x_val, y_val)
        else:
            x_val = None
            y_val = None
            using_validation_data = False

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
        range_epochs = r if verbose != 3 else tqdm(r)

        # loop over the number of epochs
        for epoch in range_epochs:
            # initialize the total loss for the epoch if verbose is 2

            epoch_metrics = {metric: 0.0 for metric in self.metrics}
            epoch_metrics['loss'] = 0.0

            val_metrics = None

            # loop over the data in batches
            total_data_used_per_epoch = 0
            generator = self.batch_generator(x, y, batch_size, shuffle)
            if verbose == 2:
                generator = tqdm(generator, total=len(x) // batch_size)

            for x_batch, y_batch in generator:

                total_data_used_per_epoch += x_batch.shape[0]

                metrics = self.train_on_batch(x_batch, y_batch)

                # update the total loss and metrics for the epoch
                for metric in epoch_metrics:
                    epoch_metrics[metric] += metrics[metric]

                if using_validation_data and verbose == 1:
                    # compute the loss and the metrics for the validation data
                    y_val_pred = self.predict(x_val)
                    val_loss = self.loss.compute_loss(y_val, y_val_pred)
                    val_metrics = {"val_" + metric: self.metrics[metric](y_val, y_val_pred)
                                   for metric in self.metrics}

                    # add val_loss to val_metrics
                    val_metrics["val_loss"] = val_loss

                # print the loss and the metrics per batch (if verbose is 1)
                if verbose == 1:
                    self.__print_results(metrics, val_metrics, "Batch",
                                         progress=total_data_used_per_epoch, total=len(x))

            # print the metrics of the epoch (if verbose is 1 or 2)
            if verbose == 1 or verbose == 2:

                # compute the average loss and metrics for the epoch
                epoch_metrics = {metric: epoch_metrics[metric] / (len(x) // batch_size) for metric in epoch_metrics}
                val_epoch_metrics = None
                if using_validation_data:
                    # compute the loss and the metrics for the validation data
                    y_val_pred = self.predict(x_val)
                    val_epoch_loss = self.loss.compute_loss(y_val, y_val_pred)
                    val_epoch_metrics = {"val_" + metric: self.metrics[metric](y_val, y_val_pred)
                                         for metric in self.metrics}

                    # add val_loss to val_metrics
                    val_epoch_metrics["val_loss"] = val_epoch_loss

                # print the metrics of the epoch
                self.__print_results(epoch_metrics, val_epoch_metrics, "Epoch",
                                     progress=epoch + 1, total=epochs, history=history)

            # save the history
            for metric in epoch_metrics:
                history[metric].append(epoch_metrics[metric] / (len(x) // batch_size))

        if verbose == 3:
            # print final results
            self.evaluate(x, y)
            if using_validation_data:
                self.evaluate(x_val, y_val, prefix="val_")
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

    def evaluate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 batch_size: int = None,
                 verbose: int = 1,
                 prefix: str = "") -> dict:
        """
        Args:
            x: the input data
            y: the labels
            batch_size: the batch size (default: len(x))
            verbose: the verbosity mode (0 or 1)

        Returns:
            The average loss and the average metrics of the model on the given data
        """
        # check the data
        x, y = self.__check_data(x, y)

        # get the batch size
        batch_size = batch_size or len(x)

        # initialize the metrics
        avg_loss = 0.0
        avg_metrics = {prefix + metric: 0.0 for metric in self.metrics}

        n_batches = len(x) // batch_size

        # loop over the batches
        for i in range(0, n_batches * batch_size, batch_size):
            # get the batch data
            y_pred = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # forward pass
            y_pred = self.predict(y_pred)

            # compute the loss for the batch
            loss = self.loss.compute_loss(y_batch, y_pred)

            # compute the metrics for the batch
            metrics = {prefix + metric: self.metrics[metric](y_batch, y_pred) for metric in self.metrics}

            # update the loss and the metrics
            avg_loss += 1/(i + 1) * (loss - avg_loss)
            for metric in self.metrics:
                avg_metrics[prefix + metric] += 1/(i + 1) * (metrics[prefix + metric] - avg_metrics[prefix + metric])

            # print the loss and the metrics per batch (if verbose is 1)
            if verbose > 0 and batch_size != len(x):
                print(f"Batch ({i // batch_size + 1}/{n_batches}) - loss: {loss}")
                print(f"\t{', '.join([prefix + f'{metric}: {metrics[prefix + metric]:.4f}' for metric in self.metrics])}")

        # print final loss and metrics
        if verbose > 0:
            print(prefix + f"loss: {avg_loss}")
            print(f"{', '.join([prefix + f'{metric}: {avg_metrics[prefix + metric]:.4f}' for metric in self.metrics])}")

        return {prefix + 'loss': avg_loss, **avg_metrics}

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
        self.concatenate(other)

    def __copy__(self):
        return deepcopy(Sequential(self.layers))