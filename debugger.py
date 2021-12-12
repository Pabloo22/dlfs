"""
Temporal file used to debug the code. Feel free to add more tests. This file
will be removed in the future.
"""
import numpy as np

from dlfs.layers import Input, Dense
from dlfs import Sequential


def test1():
    # Generating the dataset
    def f(x_0, x_1):
        return x_0 * x_1
    train_x = np.random.uniform(low=-1, high=1, size=(500, 2))
    train_y = np.array([f(x_0, x_1) for x_0, x_1 in train_x])
    test_x = np.random.uniform(low=-1, high=1, size=(100, 2))
    test_y = np.array([f(x_0, x_1) for x_0, x_1 in test_x])

    # Creating the model
    model = Sequential()
    # model.add(Input(input_shape=(2,)))
    model.add(Dense(8, activation='relu', input_shape=(2,)))
    model.add(Dense(1))

    model.summary()


if __name__ == '__main__':
    test1()
