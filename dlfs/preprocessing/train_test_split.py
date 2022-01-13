# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the train test split method function"""

import numpy as np


def train_test_split(x: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = None):
	"""Splits the dataset into train and test sets.

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
