'''
Conjugate Gradient Method with Preconditioner
(If you don't want a preconditioner, then pass an identity matrix as the fourth parameter.)
'''
import numpy as np
from gaussian_elimination import hilbert_matrix


# conjugate gradient method for solving a system of equations
# M: preconditioner
def cg_preconditioner(x0, b, A, M):
    n = len(A)  # dimension
    M_inv = np.linalg.inv(M)
    x = x0
    r = b - np.matmul(A, x0)  # residual
    d = np.matmul(M_inv, r)
    z = np.matmul(M_inv, r)
    # conjugate gradient iteration: solve for precise x in n iterations
    for i in range(n):
        # if r is zero matrix
        if np.count_nonzero(r) == 0:
            return
        alpha = np.matmul(np.transpose(r), z)/np.matmul(np.matmul(np.transpose(d), A), d)
        x += alpha*d
        r_old = r.copy()
        r -= alpha*(np.matmul(A, d))
        z_old = z.copy()
        z = np.matmul(M_inv, r)
        beta = np.matmul(np.transpose(r), z)/np.matmul(np.transpose(r_old), z_old)
        d = z+beta*d
    return x


# jacobi preconditioner: diagonal matrix of A
def jacobi_preconditioner(A):
    return np.diag(np.diag(A))


# use w=1 for Gauss-Seidel preconditioner
def ssor_preconditioner(A, w):
    # diagonal matrix of A
    D = jacobi_preconditioner(A)
    # lower triangle of a (entries below the main diagonal)
    L = np.tril(A) - D
    # upper triangle of a (entries above the main diagonal)
    U = np.triu(A) - D
    return np.matmul(np.matmul(D+w*L, np.linalg.inv(D)), D+w*U)
