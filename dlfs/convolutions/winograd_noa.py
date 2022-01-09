import numpy as np
from sympy import *

# from __future__ import print_function
# from operator import mul
# from functools import reduce

x, d, g = symbols('x,d,g')


def get_list_m(m, r):
    m_values = [0]
    i = 1
    while len(m_values) < (m + r - 1) - 1:
        if len(m_values) % 2 == 0:
            m_values.append(i)
            i += 1
        elif len(m_values) % 2 == 1:
            m_values.append(-i)
    return m_values


# get_list_m(4,3)

def get_gx_polynomial(r):
    gx = zeros(0)
    for i in range(r):
        gx = gx.row_insert(len(gx), Matrix([g * x ** i]))
    return gx.T


# gx = get_gx_polynomial(4,3)
# gx

def get_dx_polynomial(m):
    dx = zeros(0)
    for i in range(m):
        dx = dx.row_insert(len(dx), Matrix([d * x ** i]))
    return dx.T


# dx = get_dx_polynomial(4,3)
# dx

def get_a(m, r):
    A = zeros(0)
    m_values = get_list_m(m, r)
    dx = get_dx_polynomial(m)

    for i in m_values:
        A = A.row_insert(len(A), Matrix(dx.subs(x, i * -1)))
    biggest_coeff = [dx[-1]]
    A_new_row = zeros(len(dx) - 1, 1).row_insert(len(dx), Matrix(biggest_coeff)).T
    A = A.row_insert(len(A) - 1, A_new_row).subs(((x, 1), (d, 1)))
    return A


# A = get_A(4,3)
# A.T

def get_mx_polynomial(m, r):
    mx_matrix = zeros(0)
    mx_ply = 1
    m_values = get_list_m(m, r)

    for i in m_values:
        mx_matrix = mx_matrix.row_insert(len(mx_matrix), Matrix([x + i]))
        mx_ply = (x + i) * mx_ply
    return mx_ply, mx_matrix


# s,r = get_mx_polynomial(4,3)
# r

def get_b(m, r):
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


# B = get_B(4,3)
# B.T

def get_m_polynomial(m, r):
    M = zeros(0)
    m_values = get_list_m(m, r)

    mx, _ = get_mx_polynomial(m, r)

    for i in m_values:
        M = M.row_insert(len(M), Matrix([mx / (x + i)]))
    return M


# get_M_polynomial(4,3)

# n(i)(x)*m(i)(x) + N(i)(x)*M(i)(x) = 1         a*m(i)(x) + b*M(i)(x) = c
def get_n_values(m, r):
    n = zeros(0)
    _, mx = get_mx_polynomial(m, r)
    M = get_m_polynomial(m, r)

    for i in range(len(mx)):
        value = 1 / rem(M[i], mx[i])
        n = n.row_insert(len(n), Matrix([value]))
    return n


# get_N_values(4,3)

def get_gxi(m, r):
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


# get_gxi(4,3)

def get_g(m, r):
    _g = zeros(0)
    gxi = get_gxi(m, r)
    n = get_n_values(m, r)

    for i in range(len(n)):
        _g = _g.row_insert(len(_g), Matrix(gxi[i, :] * n[i]))
    _g = _g.row_insert(len(_g), gxi.row(len(n)))
    return _g


# G = get_G(4,3)
# G

filtro = [1, 2, 1]


# 2D: F(m × n, r × s)               1D: F (m, r)

def winograd_algorithm(image, m, r):  # m = number of outputs    r = size of the filter

    # G, g = symbols('G,g')

    A = get_a(m, r)
    At = A.T
    B = get_b(m, r)
    Bt = B.T
    G = get_g(m, r)
    gt = G.T

    # To make it easier
    image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

    _g = np.tile(filtro, r).reshape((r, r))
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
