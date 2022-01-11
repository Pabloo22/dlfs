import numpy as np

from dlfs.models import Sequential
from dlfs.layers import Dense, Dropout, Flatten
from dlfs.optimizers import SGD, SGDMomentum
from dlfs.losses import MSE, MAE, BinaryCrossentropy, CategoricalCrossentropy
from dlfs.activation_functions import ReLU, Sigmoid, Softmax



def get_dataset():
    def f(x_0, x_1):
        return x_0 * x_1
    train_x = np.random.uniform(low=-1, high=1, size=(500, 2))
    train_y = np.array([[f(x_0, x_1)] for x_0, x_1 in train_x])
    test_x = np.random.uniform(low=-1, high=1, size=(100, 2))
    test_y = np.array([[f(x_0, x_1)] for x_0, x_1 in test_x])

    return train_x, train_y, test_x, test_y


def get_weights(topology):
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
    import tensorflow.keras as keras

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


def test_train_on_batch():

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
        our_model.train_on_batch(x, y)
        keras_model.train_on_batch(x, y)

    our_prediction = our_model.predict(test_x)
    keras_prediction = keras_model.predict(test_x)
    assert np.allclose(our_prediction, keras_prediction)


def test_boston():
    # DATA PREPROCESSING
    from dlfs.preprocessing import train_test_split
    from sklearn.datasets import load_boston

    boston = load_boston()
    x = boston.data
    y = boston.target

    # normalize the data attributes
    x = (x - x.mean(axis=0)) / x.std(axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    model = Sequential()
    model.add(Dense(100, input_shape=(x_train.shape[1],), activation='relu'))
    model.add(Flatten())  # used just for testing purposes
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer=SGDMomentum(learning_rate=0.001), metrics=['mae'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=1, verbose=3)
    model.evaluate(x_test, y_test, prefix="test_")


def test_cancer():
    # DATA PREPROCESSING
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import MinMaxScaler
    cancer = load_breast_cancer()
    target_names = cancer.target_names
    feature_names = cancer.feature_names
    scaler = MinMaxScaler()
    scaler.fit(cancer.data)
    cancer_scaled = scaler.transform(cancer.data)
    X = cancer_scaled
    y = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = Sequential()
    model.add(Dense(100, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=SGDMomentum(learning_rate=0.001), metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=1, verbose=3)
    print(model.predict(X_test[:5]))
    print(y_test[:5])


def test_mnist_denses():
    from keras.utils import np_utils

    # TEST 1

    # LOAD DATA
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    num_train_image = X_train.shape[0]
    num_test_image = X_test.shape[0]
    image_height = X_train.shape[1]
    image_width = X_train.shape[2]
    X_train = X_train.reshape(num_train_image, image_height * image_width).astype('float32')
    X_test = X_test.reshape(num_test_image, image_height * image_width).astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    model = Sequential()
    model.add(Dense(500, input_shape=(image_height * image_width,), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=SGDMomentum(learning_rate=0.05),
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=2)


if __name__ == "__main__":
    test_boston()
