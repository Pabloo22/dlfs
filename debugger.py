"""
Temporal file used to debug the code. Feel free to add more tests. This file
will be removed in the future.
"""
import numpy as np
# import tensorflow.keras as keras

from dlfs import Sequential
from dlfs.layers import Dense
from dlfs.optimizers import SGD


def get_dataset():
    def f(x_0, x_1):
        return x_0 * x_1
    train_x = np.random.uniform(low=-1, high=1, size=(500, 2))
    train_y = np.array([[f(x_0, x_1)] for x_0, x_1 in train_x])
    test_x = np.random.uniform(low=-1, high=1, size=(100, 2))
    test_y = np.array([[f(x_0, x_1)] for x_0, x_1 in test_x])

    return train_x, train_y, test_x, test_y


def test1():
    # Generating the dataset
    train_x, train_y, test_x, test_y = get_dataset()

    # Creating the model
    model = Sequential()
    # model.add(Input(input_shape=(None, 2)))
    model.add(Dense(16, activation="relu", input_shape=(2,)))  # weight_shape: (2, 16)
    model.add(Dense(8, activation="relu"))  # weight_shape: (16, 8)
    model.add(Dense(1, activation="sigmoid"))  # weight_shape: (8, 1)

    model.summary()

    # Compiling the model
    model.compile(loss="mse", optimizer=SGD(lr=0.01))

    # Training the model
    model.fit(train_x, train_y, epochs=10, batch_size=5, verbose=2, validation_data=(test_x, test_y))

    # Evaluating the model
    print(train_x[:5])
    print(model.predict(train_x[:5]))
    print(test_y[:5])


def test2():
    # Generating the dataset
    train_x, train_y, test_x, test_y = get_dataset()

    model = keras.Sequential([
        keras.layers.Dense(16, activation="relu", input_shape=(2,)),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.summary()

    model.compile(loss="mae", optimizer=keras.optimizers.SGD(learning_rate=0.0001))

    model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=1, validation_data=(test_x, test_y))

    print(test_x[:5])
    print(model.predict(test_x[:5]))
    print(test_y[:5])


if __name__ == '__main__':
    test1()
