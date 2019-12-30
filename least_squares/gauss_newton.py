'''
Nonlinear Least Squares Methods: Gauss-Newton and Levenberg-Marquardt
'''
import numpy as np
from equation_solver.pa_lu_factorization import pa_lu_factorization, back_substitution


# Minimize r_1(x)^2+...+r_m(x)^2
# r: a system of m nonlinear equations in x1, ..., xn
# x0: initial guess for x
# m: number of iterations
# jacobian_r: the Jacobian of r
def gauss_newton(r, x0, m, jacobian_r):
    x = x0
    for k in range(m):
        A = jacobian_r(x)
        P, L, U = pa_lu_factorization(np.matmul(np.transpose(A), A))
        v = back_substitution(P, L, U, -np.matmul(np.transpose(A), r(x)))
        x += v
    return x


# Minimize r_1(x)^2+...+r_m(x)^2
# r: a system of m nonlinear equations in x1, ..., xn
# x0: initial guess for x
# m: number of iterations
# jacobian_r: the Jacobian of r
# c: regularization parameter
# when c = 0, this is equivalent to gauss-newton
def levenberg_marquardt(r, x0, m, jacobian_r, c):
    x = x0
    for k in range(m):
        A = jacobian_r(x)
        prod = np.matmul(np.transpose(A), A)
        diag_m = np.diag(np.diag(prod))
        P, L, U = pa_lu_factorization(prod+c*diag_m)
        v = back_substitution(P, L, U, -np.matmul(np.transpose(A), r(x)))
        x += v
    return x
