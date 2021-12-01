import numpy as np
from sklearn.model_selection import train_test_split
from typing import List


from layers.layer import Layer


class Sequential:
    """
    A sequential model is a linear stack of layers.

    Args:
        layers (List[Layer]): A list of layers.
        name (str): The name of the model.
    """

    def __init__(self, layers: List[Layer] = None, name: str = None):
        self.layers = layers
        self.name = name
        self.loss = None
        self.optimizer = None
        self.metrics = None
        self.trainable = True

    def add(self, layer: Layer):
        """
        Add a layer to the model
        Args:
            layer: the layer to add
        """
        self.layers.append(layer)

    def compile(self, optimizer, loss, metrics=None):
        """
        Compile the model

        Args:
            optimizer: the optimizer to use
            loss: the loss function to use
            metrics: the metrics to use
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def summary(self):
        """
        Print the summary of the model as a table
        """
        print(f"{self.name}")
        print("-" * len(self.name))
        print("Layers:")
        for layer in self.layers:
            print(f"{layer.name}")
            print("-" * len(layer.name))
            print(f"{layer.summary()}")
            print("")

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32, verbose: int = 1,
            validation_data: np.ndarray = None, validation_split: float = 0.0, shuffle: bool = True, initial_epoch=0):
        """

        Args:
            x: the input data
            y: the labels
            epochs: the number of epochs to train the model
            batch_size: the batch size
            verbose: the verbosity mode
            validation_data: the validation data
            validation_split: the validation split
            shuffle: whether to shuffle the data
            initial_epoch: the initial epoch
        """
        # check if the model has been compiled
        if self.optimizer is None or self.loss is None:
            raise ValueError("You must compile the model before training")

        if validation_split > 0:
            if validation_data is None:
                raise ValueError("If validation_split is set, validation_data must be specified")
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split)

        # initialize the history
        history = {'loss': [], 'val_loss': []}

        for metric in self.metrics:
            history[metric] = []
            history['val_' + metric] = []

        # loop over the number of epochs
        for epoch in range(initial_epoch, epochs):
            # initialize the total loss for the epoch
            epoch_loss = 0.0
            if 'acc' in self.metrics:
                epoch_acc = 0.0
            # loop over the data in batches
            for x_batch, y_batch in self.batch_generator(x, y, batch_size, shuffle):
                # get the gradients for the batch
                grads = self.optimizer.get_gradients(self, x_batch, y_batch)
                # update the parameters
                self.optimizer.update_params(self, grads)
                # compute the loss for the batch
                loss = self.loss.compute_loss(self, x_batch, y_batch)
                # update the total loss
                epoch_loss += loss
                if 'acc' in self.metrics:
                    acc = self.metrics['acc'].compute_metric(self, x_batch, y_batch)
                    epoch_acc += acc
                # update the history
                history['loss'].append(loss)
                if 'acc' in self.metrics:
                    history['acc'].append(acc)
            # compute the average loss for the epoch
            epoch_loss /= len(x)
            if 'acc' in self.metrics:
                epoch_acc /= len(x)
            # update the history
            history['val_loss'].append(epoch_loss)
            if 'acc' in self.metrics:
                history['val_acc'].append(epoch_acc)
            # print the metrics
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"\tTrain loss: {epoch_loss}")
                if 'acc' in self.metrics:
                    print(f"\tTrain acc: {epoch_acc}")
                if validation_data is not None:
                    val_loss = self.loss.compute_loss(self, x_val, y_val)
                    val_acc = self.metrics['acc'].compute_metric(self, x_val, y_val)
                    print(f"\tValidation loss: {val_loss}")
                    print(f"\tValidation acc: {val_acc}")
                print("")

        return history

    @staticmethod
    def batch_generator(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
        """

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

    def predict(self, x: np.ndarray, batch_size: int = 32, verbose: int = 1):
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
        y_pred = []

        for i in range(len(x)):
            # TODO: get the prediction for the sample
            pass

        return np.array(y_pred)
