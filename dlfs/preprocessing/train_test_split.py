import numpy as np


def train_test_split(x: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = None):
	"""
	Split the dataset into train and test sets.
	Args:
		x: input data
		y: labels
		test_size: the proportion of the dataset to include in the test split
		random_state: the seed used by the random number generator

	Returns:
		x_train: training data
		y_train: training labels
		x_test: testing data
		y_test: testing labels
	"""
	train = x.shape[0] - int(x.shape[0] * test_size)
	arr = np.arange(x.shape[0])
	np.random.RandomState(seed=random_state).shuffle(arr)
	return x[arr[:train]], x[arr[train:]], y[arr[:train]], y[arr[train:]]
