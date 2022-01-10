import numpy as np
from sympy import *

# Source: Fast Algorithms for Convolutional Neural Networks
# By: Andrew Lavin and Scott Gray Nervana Systems
# (https://openaccess.thecvf.com/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)

# Appendices
# (https://openaccess.thecvf.com/content_cvpr_2016/supplemental/Lavin_Fast_Algorithms_for_2016_CVPR_supplemental.pdf)

x, d, g = symbols('x,d,g')


def get_list_m(m: int, r: int):
    """
    This function generates negative and positive numbers in ascend order used for the equation m(x).
        m(x) = x(x-1)(x+1)(x-2)...(x-∞)

    Args:
        m (int): The size of the output
        r (int): The size of the filter

     Returns:
         A list with the numbers chosen to create the polynomial m(x).
         [-1,1,-2,...]
    """
    m_values = [0]
    i = 1
    while len(m_values) < (m + r - 1) - 1:

        if len(m_values) % 2 == 0:
            m_values.append(i)
            i += 1

        elif len(m_values) % 2 == 1:
            m_values.append(-i)

    return m_values


def get_gx_polynomial(r: int):
    """
    This function generates a polynomial g(x) of degree (r-1).

    Args:
        r (int): The size of the filter

     Returns:
         A matrix of dimension (1,r) with each term in each position.
    """

    gx = zeros(0)
    for i in range(r):
        gx = gx.row_insert(len(gx), Matrix([g * x ** i]))
    return gx.T


def get_dx_polynomial(m: int):
    """
    This function generates a polynomial d(x) of degree (m-1).

    Args:
        m (int): The size of the output

     Returns:
         A matrix of dimension (1,d) with each term in each position.
    """
    dx = zeros(0)
    for i in range(m):
        dx = dx.row_insert(len(dx), Matrix([d * x ** i]))
    return dx.T


def get_a(m: int, r: int):
    """
    This function generates a linear system calculating d(i)(x) = d(x) mod m(i) for each i = 0, 1, _, m, len(m_values).
        d(0)(x) = d(x) mod m(0) = d0
        d(1)(x) = d(x) mod m(1) = d0 + d1 + ... + d(m-1)
        .
        .
        d(i)(x) = d(x) mod m(i) = d(i)
        d(len(m_values))(x) = d(x) mod m(len(m_values)) = d(x)[-1]

    Args:
        m (int): The size of the output
        r (int): The size of the filter

     Returns:
         The transformation of the linear system into a matrix of dimension (m+r-1, m).
    """
    A = zeros(0)
    m_values = get_list_m(m, r)
    dx = get_dx_polynomial(m)

    for i in m_values:
        A = A.row_insert(len(A), Matrix(dx.subs(x, i * -1)))

    biggest_coeff = [dx[-1]]
    A_new_row = zeros(len(dx) - 1, 1).row_insert(len(dx), Matrix(biggest_coeff)).T
    A = A.row_insert(len(A) - 1, A_new_row).subs(((x, 1), (d, 1)))
    return A


def get_mx_polynomial(m: int, r: int):
    """
    This function takes the list of numbers from 'get_list_m' and generates que equation for m(x).
        m(x) = x(x-1)(x+1)(x-2)...(x-∞)

    Args:
        m (int): The size of the output
        r (int): The size of the filter

     Returns:
         mx_ply: A representation of m(x) as a polynomial.
         mx_matrix: A representation of m(x) as a matrix which each term in each position.
    """
    mx_matrix = zeros(0)
    mx_ply = 1
    m_values = get_list_m(m, r)

    for i in m_values:
        mx_matrix = mx_matrix.row_insert(len(mx_matrix), Matrix([x + i]))
        mx_ply = (x + i) * mx_ply
    return mx_ply, mx_matrix


def get_b(m: int, r: int):
    """
    This function generates a linear system composed with the calculating M(i)(x) = m(x)/m(i)(x)
    for each i = 0, 1, _, m and the polynomial m(x).
        M(0)(x) = m(x)/m(0)(x)
        M(1)(x) = m(x)/m(1)(x)
        .
        .
        M(i)(x) = m(x)/m(i)(x)
        .
        m(x) = x(x-1)(x+1)(x-2)...


    Args:
        m (int): The size of the output
        r (int): The size of the filter

     Returns:
         The transformation of the linear system into a square matrix of order (m+r-1).
    """
    B = zeros(0)
    m_values = get_list_m(m, r)
    mx, t = get_mx_polynomial(m, r)

    for i in m_values:
        M = Poly(mx / (x + i))
        M_coeffs = M.all_coeffs()[::-1]
        M_coeffs.append(0)
        B = B.row_insert(len(B), Matrix(M_coeffs).T)
    B_last_row = Poly(expand(mx)).all_coeffs()[::-1]
    B = B.T.col_insert(len(B), Matrix(B_last_row))
    return B


def get_m_polynomial(m: int, r: int):
    """
    This function generates a linear system composed with the calculating
    M(i)(x) = m(x)/m(i)(x) for each i = 0, 1, _, m and the polynomial m(x).
        M(0)(x) = m(x)/m(0)(x)
        M(1)(x) = m(x)/m(1)(x)
        .
        .
        M(i)(x) = m(x)/m(i)(x)
        .
        m(x) = x(x-1)(x+1)(x-2)...

    Args:
        m (int): The size of the output
        r (int): The size of the filter

     Returns:
         A matrix of dimension (m+r-1, 1) with each polynomial.
    """
    M = zeros(0)
    m_values = get_list_m(m, r)

    mx, _ = get_mx_polynomial(m, r)

    for i in m_values:
        M = M.row_insert(len(M), Matrix([mx / (x + i)]))
    return M


# n(i)(x)*m(i)(x) + N(i)(x)*M(i)(x) = 1         a*m(i)(x) + b*M(i)(x) = c
def get_n_values(m: int, r: int):
    """
    This function applies the Chinese Remainder Theorem.

    Args:
        m (int): The size of the output
        r (int): The size of the filter

     Returns:
         A matrix of dimension (m+r-1, 1) with value.
    """
    n = zeros(0)
    _, mx = get_mx_polynomial(m, r)
    M = get_m_polynomial(m, r)

    for i in range(len(mx)):
        value = 1 / rem(M[i], mx[i])
        n = n.row_insert(len(n), Matrix([value]))
    return n


def get_gxi(m: int, r: int):
    """
    Description:
        This function generates a linear system calculating g(i)(x) = g(x) mod m(i)
        for each i = 0, 1, _, m, len(m_values).
            g(0)(x) = g(x) mod m(0) = g0
            g(1)(x) = g(x) mod m(1) = g0 + g1 + ... + g(m-1)
            .
            .
            g(i)(x) = g(x) mod m(i) = ....
            g(len(m_values))(x) = g(x) mod m(len(m_values)) = g(x)[-1]

    Args:
        m (int): The size of the output
        r (int): The size of the filter

     Returns:
         The transformation of the linear system into a matrix of dimension (m+r-1, r).
    """
    gxi = zeros(0)
    m_values = get_list_m(m, r)
    mx, _ = get_mx_polynomial(m, r)
    gx = get_gx_polynomial(r)

    for i in m_values:
        gxi = gxi.row_insert(len(gxi), Matrix(gx.subs(x, i * -1)))
    biggest_coeff = [gx[len(gx) - 1]]
    gxi_new_row = zeros(len(gx) - 1, 1).row_insert(len(gx), Matrix(biggest_coeff)).T
    gxi = gxi.row_insert(len(gxi) - 1, gxi_new_row).subs(((x, 1), (g, 1)))
    return gxi


def get_g(m: int, r: int):
    """
    Description:
        This function calculates n.T @ gxi.

    Args:
        m (int): The size of the output
        r (int): The size of the filter

     Returns:
         The resulting matrix of the multiplication which is of dimension (m+r-1, r).
    """
    _g = zeros(0)
    gxi = get_gxi(m, r)
    n = get_n_values(m, r)

    for i in range(len(n)):
        _g = _g.row_insert(len(_g), Matrix(gxi[i, :] * n[i]))
    _g = _g.row_insert(len(_g), gxi.row(len(n)))
    return _g


def winograd_algorithm(image: np.ndarray, m: int, r: int):  # m = number of outputs    r = size of the filter
    """
    Description:
        This function calculates the convolution of the given image.

    Args:
        image (np.ndarray) : The kernel of the given image.
        m (int): The size of the output
        r (int): The size of the filter

     Returns:
         The resulting convolution.
    """

    A = get_a(m, r)
    At = A.T
    B = get_b(m, r)
    Bt = B.T
    G = get_g(m, r)
    gt = G.T

    # To make it easier
    image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

    _g = np.ones((r, r))
    input_size = m + r - 1

    number_of_inputs = image.shape[0] - input_size + 1
    result = np.zeros(1)

    # A minimal 1d algorithm is nested with itself to obtain a minimal 2d algorithm. F (m x m, r x r)
    for i in range(image.shape[0] - input_size + 1):
        for j in range(image.shape[1] - input_size + 1):
            _d = image[i:i + input_size, j:j + input_size]

            # 4x3 3x3 = 4x3 3x4 = 4x4
            s11 = np.dot(G, _g)
            s12 = np.dot(s11, gt)

            # 4x4 4x4 = 4x4 4x4 = 4x4
            s21 = np.dot(B, _d)
            s22 = np.dot(s21, Bt)

            s3 = np.multiply(s12, s22)
            s4 = np.dot(At, s3)
            Y = np.dot(s4, A)
            convolved = Y.sum()

            result = np.hstack([result, convolved])

    result = Matrix(np.delete(result, 0, axis=0).reshape((number_of_inputs, number_of_inputs)))
    return result


if __name__ == "__main__":
    img = np.ones((7, 7), dtype=int)
    winograd_algorithm(img, 3, 3)
