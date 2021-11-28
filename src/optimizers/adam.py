import numpy as np

from base import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.

    Args:
        lr (float): learning rate
        beta1 (float): first moment decay
        beta2 (float): second moment decay
        epsilon (float): epsilon for numerical stability
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

        super(Adam, self).__init__(lr)
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__epsilon = epsilon
        self.__m = None
        self.__v = None
        self.__t = 0

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

    def update(self, parameters: np.ndarray, gradients: np.ndarray):
        """
        Update the parameters of the model using the Adam optimization algorithm.

        Args:
            parameters: Parameters of the model.
            gradients: Gradients of the model.
        """
        if self.__m is None:
            self.__m = np.zeros_like(parameters)
            self.__v = np.zeros_like(parameters)

        self.__t += 1
        self.__m = self.__beta1 * self.__m + (1 - self.__beta1) * gradients
        self.__v = self.__beta2 * self.__v + (1 - self.__beta2) * np.square(gradients)
        m_hat = self.__m / (1 - self.__beta1 ** self.__t)
        v_hat = self.__v / (1 - self.__beta2 ** self.__t)
        parameters -= self.__learning_rate * m_hat / (np.sqrt(v_hat) + self.__epsilon)
