# Source: On Improving the Numerical Stability of Winograd Convolutions
# By: Kevin Vincent and Kevin J. Stephano and Michael A. Frumkin and Boris Ginsburg and Julien Demouth
# (https://openreview.net/pdf?id=H1ZaRZVKg)

import numpy as np
from typing import Union, List, Tuple
import tensorly
from skimage.util import view_as_blocks

np.set_printoptions(formatter={
    'float': lambda x: "{0:0.3f}".format(x)
})


def vandermonde_matrix(b: int, points: List[Tuple[Union[int, float]]]) -> np.ndarray:
    """
    This is equivalent to V_{axb} on the document.
    This function returns a trimmed Vandermonde Matrix of size len(points) image b

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


# commented to avoid duplicate code warnings
# def gen_points(num_points: int) -> List[Tuple[float, int]]:
#     """
#     This function generates homogeneous coordinates, starting from (0, 1), then, (1/2, 1), then (-1/2, 1),
#     and the infinite point (0, 1).
#
#         [(f_0, g_0), ..., (f_a-1, g_a-1)]
#
#     Args:
#         num_points (int): The desired number of points
#
#     Returns:
#         list[tuple[Union[int, float]]]: List of homogeneous coordinates
#     """
#     points = [(0, 1)]
#     if num_points % 2:
#         points.extend([(y / 2, 1) for x in range(1, num_points // 2) for y in (x, -x)])
#         points.extend([(num_points // 2 / 2, 1), (1, 0)])
#     else:
#         points.extend([(y / 2, 1) for x in range(1, num_points // 2) for y in (x, -x)])
#         points.append((1, 0))
#     return points


def winograd_get_matrices(m: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This is equivalent to F(m, n) on the paper.
    This function returns the matrices necessary for calculating the Winograd convolution given the dimension of the filter and the dimension of the output. With those, we can infer the size of the input.
    The usage of these matrices depends on the dimension of the convolution. If we denote image as the signal and w as the filter:
    - If it is 1D: F(m, n) = image * w = Y.T @ [(X.T @ image) * (W @ w)]
    - If it is 2D: F(m image i, n image j) = image * w = Y.T @ [(X.T @ image @ X') * (W @ w @ W'.T)] @ Y'
    In the latter case, if m = i and n = j, then Y = Y', X = X', W = W'. If not, then you would have to calculate both F(m, n) to get Y, X, W and F(i, j) to get Y', X', W'.

    Args:
        m (int): The size of the output
        n (int): The size of the filter

    Returns:
        A tuple with the aforementioned Y.T, X, W in this order.
    """
    num_points = m + n - 1
    points = gen_points(num_points)
    Y = vandermonde_matrix(m, points)
    X = np.linalg.inv(vandermonde_matrix(num_points, points))
    W = vandermonde_matrix(n, points)
    return Y.T, X, W


def winograd_chunk_2D(chunk, filter, YT, X1, W1, Y, X2, W2):
    return YT @ ((X1.T @ chunk @ X2) * (W1 @ filter @ W2.T)) @ Y.T


def winograd_chunk_3D(chunk, filter, winograd_matrices):
    part1 = tensorly.tenalg.multi_mode_dot(chunk, [t[1] for t in winograd_matrices], list(range(chunk.ndim)))
    part2 = tensorly.tenalg.multi_mode_dot(filter, [t[2] for t in winograd_matrices], list(range(filter.ndim)))
    part3 = part1 * part2
    return tensorly.tenalg.multi_mode_dot(part3, [t[0] for t in winograd_matrices], list(range(part3.ndim)))


def winograd_convolution(source, filter):
    if not source.ndim == filter.ndim:
        raise IndexError
    already_calculated = []
    winograd_matrices = []
    if source.ndim == 1:
        out_dim = source.shape[0] - filter.shape[0] + 1
        YT, X, W = winograd_get_matrices(out_dim, filter.shape[0])
        out = YT @ ((X.T @ source @ X) * (W @ filter))
    elif source.ndim == 2:
        out_dim1 = source.shape[0] - filter.shape[0] + 1
        YT, X1, W1 = winograd_get_matrices(out_dim1, filter.shape[0])
        out_dim2 = source.shape[1] - filter.shape[1] + 1
        if out_dim1 != out_dim2 or filter.shape[0] != filter.shape[1]:
            Y, X2, W2 = winograd_get_matrices(out_dim2, filter.shape[1])
        else:
            Y, X2, W2 = YT, X1, W1
        # out = YT @ ((X1.T @ source @ X2) * (W1 @ filter @ W2.T)) @ Y.T
    else:
        for i, j in zip(source.shape, filter.shape):
            out_dim = i - j + 1
            if not (out_dim, j) in already_calculated:
                already_calculated.append((out_dim, j))
                winograd_matrices.append(winograd_get_matrices(out_dim, j))
            else:
                winograd_matrices.append(winograd_matrices[already_calculated.index((out_dim, j))])

        # part1 = tensorly.tenalg.multi_mode_dot(source, [t[1] for t in winograd_matrices], list(range(source.ndim)))
        # part2 = tensorly.tenalg.multi_mode_dot(filter, [t[2] for t in winograd_matrices], list(range(filter.ndim)))
        # part3 = part1 * part2
        # out = tensorly.tenalg.multi_mode_dot(part3, [t[0] for t in winograd_matrices], list(range(part3.ndim)))
    #return out


def winograd_convolution_with_blocks(source, filter, blocksize=None):
    if not source.ndim == filter.ndim:
        raise IndexError
    already_calculated = []
    winograd_matrices = []
    if source.ndim == 2:
        if blocksize == None:
            num_of_passes = (source.shape[0] - filter.size[0], source.shape[1] - filter.size[1])
            shape1 = -1
            shape2 = -1
            for i in range(source.shape[0] + 1, source.shape[0]):
                if num_of_passes[0] % shape1 == 0:
                    shape1 = i
                    break
            for i in range(source.shape[1] + 1, source.shape[1]):
                if num_of_passes[1] % shape2 == 0:
                    shape2 = i
                    break
            blocksize = np.array((shape1, shape2))
        num_of_blocks = np.array((source.shape[0]/blocksize[0], source.shape[1]/blocksize[1]))
        if np.prod(num_of_blocks).dtype != int:
            raise ValueError
        block_output_size = blocksize + 1 - filter.size
        output_size = block_output_size * num_of_blocks

if __name__ == "__main__":
    '''Y, X, W = winograd_get_matrices(3, 3)
    print('Y:\n', Y)
    print('X:\n', X)
    print('W:\n', W)'''
    test_image = np.arange(4 * 4).reshape(4, 4)
    test_filter = np.arange(2 * 2).reshape(2, 2)
    print(winograd_convolution(test_image, test_filter))
