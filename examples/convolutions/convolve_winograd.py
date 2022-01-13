'''
Sources
On Improving the Numerical Stability of Winograd Convolutions** by Kevin Vincent and Kevin J. Stephano and Michael A. Frumkin and Boris Ginsburg and Julien Demouth
(https://openreview.net/pdf?id=H1ZaRZVKg)
'''
import numpy as np
from typing import Union, Tuple, List
import tensorly


def V(b: int, points: List[Tuple[Union[int, float]]]) -> np.ndarray:
    """
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
    return np.array([[i[0] ** j * i[1] ** (b - j - 1) for j in range(b)] for i in points], dtype=float)


def gen_points(num_points: int, half) -> List[Tuple[Union[int, float]]]:
    """
    This function generates homogeneous coordinates, starting from (0, 1), then, (1/2, 1), then (-1/2, 1), and the infinite point (0, 1).

        [(f_0, g_0), ..., (f_a-1, g_a-1)]

    Args:
        num_points (int): The desired number of points
        half (bool): Sets if you want 1/2 or 1

    Returns:
        list[tuple[Union[int, float]]]: List of homogeneous coordinates
    """
    points = [(0, 1)]
    if half:
        if num_points % 2:
            points.extend([(y / 2, 1) for x in range(1, num_points // 2) for y in (x, -x)])
            points.extend([(num_points // 2 / 2, 1), (1, 0)])
            return points
        else:
            points.extend([(y / 2, 1) for x in range(1, num_points // 2) for y in (x, -x)])
            points.append((1, 0))
            return points

    else:
        if num_points % 2:
            points.extend([(y, 1) for x in range(1, num_points // 2) for y in (x, -x)])
            points.extend([(num_points // 2, 1), (1, 0)])
        else:
            points.extend([(y, 1) for x in range(1, num_points // 2) for y in (x, -x)])
            points.append((1, 0))
    return points


def F(m: int, n: int, half: bool = False) -> Tuple[np.ndarray]:
    """
    This function returns the matrices neccesary for calculating the Winograd convolution given the dimension of the filter and the dimension of the output. With those, we can infer the size of the input.
    The usage of these matrices depends on the dimension of the convolution. If we denote x as the signal and w as the filter:
    - If it is 1D: F(m, n) = x * w = Y.T @ [(X.T @ x) * (W @ w)]
    - If it is 2D: F(m x i, n x j) = x * w = Y.T @ [(X.T @ x @ X') * (W @ w @ W'.T)] @ Y'
    - If it is 3D or more: F(mxixk, nxjxl) = [(x x_1 x_2 x_3 X) * (w x_1 x_2 x_3 W)] x_1 x_2 x_3 Y
    In the latter case, if m = i and n = j, then Y = Y', X = X', W = W'. If not, then you would have to calculate both F(m, n) to get Y, X, W and F(i, j) to get Y', X', W'.

    Args:
        m (int): The size of the output
        n (int): The size of the filter
        half (bool): Sets if you want 1/2 or 1

    Returns:
        tuple[np.ndarray[Union[int, float]]]: A tuple with the aforementioned Y.T, X, W in this order.
    """
    num_points = m + n - 1
    points = gen_points(num_points, half)
    Y = V(m, points)
    X = np.linalg.inv(V(num_points, points))
    W = V(n, points)
    return Y.T, X, W


def convolve_winograd(source: np.ndarray, filter: np.ndarray, half: bool = False) -> np.ndarray:
    if source.ndim != filter.ndim:
        raise ValueError(
            f"Dimenion of the souce({source.ndim}) is different from the dimension of the kernel({filter.ndim})")
    already_calculated = []
    winograd_matrices = []
    if source.ndim == 1:
        out_dim = source.shape[0] - filter.shape[0] + 1
        YT, X, W = F(out_dim, filter.shape[0], half)
        out = YT @ ((X.T @ source @ X) * (W @ filter))
    elif source.ndim == 2:
        out_dim1 = source.shape[0] - filter.shape[0] + 1
        YT, X1, W1 = F(out_dim1, filter.shape[0], half)
        out_dim2 = source.shape[1] - filter.shape[1] + 1
        if out_dim1 != out_dim2 or filter.shape[0] != filter.shape[1]:
            Y, X2, W2 = F(out_dim2, filter.shape[1], half)
        else:
            Y, X2, W2 = YT, X1, W1
        out = YT @ ((X1.T @ source @ X2) * (W1 @ filter @ W2.T)) @ Y.T
    else:
        for i, j in zip(source.shape, filter.shape):
            out_dim = i - j + 1
            if not (out_dim, j) in already_calculated:
                already_calculated.append((out_dim, j))
                winograd_matrices.append(F(out_dim, j, half))
            else:
                winograd_matrices.append(winograd_matrices[already_calculated.index((out_dim, j))])

        part1 = tensorly.tenalg.multi_mode_dot(source, [t[1] for t in winograd_matrices], list(range(source.ndim)))
        part2 = tensorly.tenalg.multi_mode_dot(filter, [t[2] for t in winograd_matrices], list(range(filter.ndim)))
        part3 = part1 * part2
        out = tensorly.tenalg.multi_mode_dot(part3, [t[0] for t in winograd_matrices], list(range(part3.ndim)))
    return out


if __name__ == '__main__':
    print(F(2, 3))
    print(convolve_winograd(np.random.uniform(-1, 1, (5)), np.random.uniform(-1, 1, (3))))
    print(convolve_winograd(np.random.uniform(-1, 1, (5, 5)), np.random.uniform(-1, 1, (3, 3))))
    print(convolve_winograd(np.random.uniform(-1, 1, (5, 5, 5)), np.random.uniform(-1, 1, (3, 3, 3))))
