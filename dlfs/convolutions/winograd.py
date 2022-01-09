from dlfs.convolutions import Convolution

# Source: On Improving the Numerical Stability of Winograd Convolutions
# By: Kevin Vincent and Kevin J. Stephano and Michael A. Frumkin and Boris Ginsburg and Julien Demouth
# (https://openreview.net/pdf?id=H1ZaRZVKg)

import numpy as np
from typing import Union, List, Tuple

np.set_printoptions(formatter={
    'float': lambda x: "{0:0.3f}".format(x)
})


def vandermonde_matrix(b: int, points: List[Tuple[Union[int, float]]]) -> np.ndarray:
    """
    This is equivalent to V_{axb} on the document.
    This function returns a trimmed Vandermonde Matrix of size len(points) x b

        f_0^0 * g_0^b-1       f_0^1 * g_0^b-2       ... f_0^b-1 * g_0^0
        .
        .
        f_a-1^0 * g_a-1^b-1   f_a-1^1 * g_a-1^b-2   ... f_a-1^b-1 * g_a-1^0

    Arguments:
        b (int) : One of the dimensions of the matrix
        points (list[tuple[Union[int, float]]]) : List of homogeneous coordinates
    Returns:
        V (np.ndarray[Union[int, float]]) : The trimmed Vandermonde matrix
    """
    return np.array([[i[0] ** j * i[1] ** (b - j - 1) for j in range(b)] for i in points])


def gen_points(num_points: int) -> List[Tuple[float, int]]:
    """
    This function generates homogeneous coordinates, starting from (0, 1), then, (1/2, 1), then (-1/2, 1), and the infinite point (0, 1).

        [(f_0, g_0), ..., (f_a-1, g_a-1)]

    Args:
        num_points (int): The desired number of points

    Returns:
        list[tuple[Union[int, float]]]: List of homogeneous coordinates
    """
    points = [(0, 1)]
    if num_points % 2:
        points.extend([(y / 2, 1) for x in range(1, num_points // 2) for y in (x, -x)])
        points.extend([(num_points // 2 / 2, 1), (1, 0)])
    else:
        points.extend([(y / 2, 1) for x in range(1, num_points // 2) for y in (x, -x)])
        points.append((1, 0))
    return points


def winograd_get_matrices(m: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This is equivalenT to F(m, n) on the paper.
    This function returns the matrices neccesary for calculating the Winograd convolution given the dimension of the filter and the dimension of the output. With those, we can infer the size of the input.
    The usage of these matrices depends on the dimension of the convolution. If we denote x as the signal and w as the filter:
    - If it is 1D: F(m, n) = x * w = Y.T @ [(X.T @ x) * (W @ w)]
    - If it is 2D: F(m x i, n x j) = x * w = Y.T @ [(X.T @ x @ X') * (W @ w @ W'.T)] @ Y'
    In the latter case, if m = i and n = j, then Y = Y', X = X', W = W'. If not, then you would have to calculate both F(m, n) to get Y, X, W and F(i, j) to get Y', X', W'.

    Args:
        m (int): The size of the output
        n (int): The size of the filter

    Returns:
        A tuple with the aforementioned Y, X, W in this order.
    """
    num_points = m + n - 1
    points = gen_points(num_points)
    Y = vandermonde_matrix(m, points)
    X = np.linalg.inv(vandermonde_matrix(num_points, points))
    W = vandermonde_matrix(n, points)
    return Y, X, W


if __name__ == "__main__":
    Y, X, W = winograd_get_matrices(3, 3)
    print('Y:\n', Y)
    print('X:\n', X)
    print('W:\n', W)
