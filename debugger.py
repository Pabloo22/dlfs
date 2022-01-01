"""
Temporal file used to debug the code. Feel free to add more tests. This file
will be removed in the future.
"""
import numpy as np
# import tensorflow.keras as keras

from dlfs import Sequential
from dlfs.layers import Dense
from dlfs.optimizers import SGD
from dlfs.losses import MSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy
from dlfs.activation_functions import ReLU, Sigmoid, Softmax


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


def test3():
    # Example of backpropagation
    # batch_size = 2

    # input_shape = (batch_size, 2)
    x = np.array([[0.5, 0.5],
                  [0.5, -0.5]])
    # rows correspond to different samples
    # columns correspond to different features
    print("x:\n", x)

    # target
    y_true = np.array([[0.25],
                       [-0.25]])

    # create seed
    np.random.seed(2)

    # BUILDING THE MODEL
    # 16 neurons
    weights_1 = np.random.uniform(low=-1, high=1, size=(2, 16))
    bias_1 = np.zeros((1, 16))
    # 8 neurons
    weights_2 = np.random.uniform(low=-1, high=1, size=(16, 8))
    bias_2 = np.zeros((1, 8))
    # 1 neuron
    weights_3 = np.random.uniform(low=-1, high=1, size=(8, 1))
    bias_3 = np.zeros((1, 1))

    # FORWARD PASS
    relu = ReLU()

    # layer 1
    z_1 = x @ weights_1 + bias_1
    a_1 = relu(z_1)
    print("a_1:\n", a_1)

    # layer 2
    # the output shape of the second layer is (16, 8)
    z_2 = a_1 @ weights_2 + bias_2
    a_2 = relu(z_2)  # shape: (2, 8)
    print("a_2:\n", a_2)

    # layer 3
    # the output shape of the third layer is (8, 1)
    y_pred = a_2 @ weights_3 + bias_3  # shape: (2, 1)
    print("y_pred:\n", y_pred)

    # BACKWARD PASS
    # -----------------
    gradient_1 = MSE.gradient(y_true, y_pred)  # gradient of loss function. shape: (2, 1)
    print("gradient_1:\n", gradient_1)
    gradient_2 = a_2 * gradient_1  # shape: (2, 8)
    print("gradient_2:\n", gradient_2)
    gradient_3 = ...
    print("gradient_3:\n", gradient_3)


def test_loss_functions():

    y_true = np.array([[1],
                       [0],
                       [1]])
    y_pred = np.array([[0.5],
                       [0.05],
                       [0.8]])

    mae = MAE()
    print("MAE:")
    print(mae(y_true, y_pred))
    print(mae.gradient(y_true, y_pred))

    mse = MSE()
    print("MSE:")
    print(mse(y_true, y_pred))
    print(mse.gradient(y_true, y_pred))

    bce = BinaryCrossEntropy()
    print("BinaryCrossEntropy:")
    print(bce(y_true, y_pred))
    print(bce.gradient(y_true, y_pred))


if __name__ == '__main__':
    test1()
