import numpy as np
import tensorflow.keras as keras

from dlfs import Sequential
from dlfs.layers import Dense


def get_dataset():
    def f(x_0, x_1):
        return x_0 * x_1
    train_x = np.random.uniform(low=-1, high=1, size=(500, 2))
    train_y = np.array([[f(x_0, x_1)] for x_0, x_1 in train_x])
    test_x = np.random.uniform(low=-1, high=1, size=(100, 2))
    test_y = np.array([[f(x_0, x_1)] for x_0, x_1 in test_x])

    return train_x, train_y, test_x, test_y


def get_weights(topology: list[int]):
    """
    Get the weights of a model.
    """
    weights = []
    for i in range(len(topology) - 1):
        weights.append(np.random.uniform(low=-1, high=1, size=(topology[i], topology[i + 1])))
    return weights


def create_our_model(weights):
    model = Sequential()

    dense1 = Dense(16, activation="relu", input_shape=(2,))
    dense1.initialize((None, 2), weights=weights[0], bias=np.ones((1, 16)))
    model.add(dense1)
    dense2 = Dense(8, activation="relu")
    dense2.initialize((None, 16), weights=weights[1], bias=np.ones((1, 8)))
    model.add(dense2)
    dense3 = Dense(1, activation="linear")
    dense3.initialize((None, 8), weights=weights[2], bias=np.ones((1, 1)))
    model.add(dense3)

    return model


def create_keras_model(weights):
    model = keras.Sequential()

    dense1 = keras.layers.Dense(16, activation="relu", input_shape=(2,), name="dense1")
    model.add(dense1)
    dense2 = keras.layers.Dense(8, activation="relu", name="dense2")
    model.add(dense2)
    dense3 = keras.layers.Dense(1, activation="linear", name="dense3")
    model.add(dense3)

    model.set_weights(weights)

    return model


def test_forward_pass():
    """
    Test the forward pass of the model.
    """
    train_x, train_y, test_x, test_y = get_dataset()

    topology = [2, 16, 8, 1]
    weights = get_weights(topology)
    biases = [np.ones((16,)), np.ones((8,)), np.ones((1,))]
    keras_weights = [*zip(weights, biases)]
    result = []
    for w, b in keras_weights:
        result.append(w)
        result.append(b)

    keras_weights = result

    our_model = create_our_model(weights)
    keras_model = create_keras_model(keras_weights)

    batch_generator = Sequential.batch_generator(train_x, train_y, batch_size=10)

    for x, y in batch_generator:
        our_prediction = our_model.predict(x)
        keras_prediction = keras_model.predict(x)
        assert np.allclose(our_prediction, keras_prediction)


if __name__ == "__main__":
    test_forward_pass()