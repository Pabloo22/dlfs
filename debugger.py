"""
Temporal file used to debug the code. Feel free to add more tests. This file
will be removed in the future.
"""
import numpy as np

from dlfs import Sequential
from dlfs.layers import Dense
from dlfs.losses import MSE
from dlfs.optimizers import SGD


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

    # Compiling the model
    model.compile(loss=MSE(), optimizer=SGD(lr=0.01))

    # Training the model
    model.fit(train_x, train_y, epochs=2, batch_size=10, verbose=1, validation_data=(test_x, test_y))


# def test2():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
#         tf.keras.layers.Dense(1)
#     ])
#
#     model.summary()


if __name__ == '__main__':
    test1()
