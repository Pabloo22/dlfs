import numpy as np

from dlfs.activation_functions import Sigmoid, ReLU, Softmax
from dlfs.losses import MSE


def backpropagation_example():

    x = np.array([[.05, .1],
                  [.1, .25],
                  [.25, .3]])  # shape: (3, 2)

    y = np.array([[.01, .99],
                  [.99, .01],
                  [.99, .99]])  # shape: (3, 2)

    # BUILDING THE MODEL
    weights_h = np.array([[.15, .25],
                          [.2, .30]])

    bias_1 = np.array([.35, .35])

    weights_o = np.array([[.40, .50],
                          [.45, .55]])

    bias_2 = np.array([.60, .60])

    sigmoid = Sigmoid()

    # FORWARD PASS
    # -----------------
    # layer 1
    net_h = np.tensordot(x, weights_h, axes=1) + bias_1
    out_h = sigmoid(net_h)
    print("out_h:\n", out_h)

    # layer 2
    net_o = np.tensordot(out_h, weights_o, axes=1) + bias_2
    out_o = sigmoid(net_o)
    print("out_o:\n", out_o)

    # COMPUTING THE LOSS
    # -----------------
    # the loss function is the mean squared error
    mse = MSE()
    loss = mse(y, out_o)
    print("loss:\n", loss)

    # BACKWARD PASS
    # -----------------
    # layer 2
    d_output = mse.gradient(y, out_o)
    print("d_output:\n", d_output)
    d_net_o = sigmoid.gradient(net_o)
    print("d_net_o:\n", d_net_o)
    d_weights_o = np.tensordot(out_h.T, d_output * d_net_o, axes=1)
    print("d_weights_o:\n", d_weights_o)


if __name__ == '__main__':
    backpropagation_example()
