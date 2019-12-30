'''
Generalized Minimum Residual Method (GMRES)
'''
import numpy as np
from qr_factorization import householder
from equation_solver.pa_lu_factorization import pa_lu_factorization, back_substitution


# Basic GMRES
# Solve the system Ax = b
# x0: initial guess for x
# m: number of iterations
def gmres(A, b, x0, m):
    x = x0
    r = b - np.matmul(A, x0)
    q = np.zeros((len(r), m+1))
    h = np.zeros((m+1, m+1))
    q[:, 0] = r/np.linalg.norm(r)
    for k in range(m):
        y = np.matmul(A, q[:, k])
        for j in range(k):
            h[j, k] = np.matmul(np.transpose(q[:, j]), y)
            y -= h[j, k] * q[:, j]
        h[k+1, k] = np.linalg.norm(y)
        if h[k+1, k] != 0:
            q[:, k+1] = y/h[k+1, k]
        b1 = np.zeros(k+2)
        b1[0] = np.linalg.norm(r)
        Q, R = householder(h[:(k+2), :(k+1)])
        P, L, U = pa_lu_factorization(R[:(k+1), :(k+1)])
        c = back_substitution(P, L, U, np.matmul(np.transpose(Q), b1)[:(k+1)])
        x = np.matmul(q[:,:(k+1)], c) + x0
        if h[k+1, k] == 0:
            return x
    return x


# Preconditioned GMRES
# Solve the system Ax = b
# x0: initial guess for x
# m: number of iterations
# M: preconditioner
def precond_gmres(A, b, x0, m, M):
    x = x0
    P_M, L_M, U_M = pa_lu_factorization(M)
    r = back_substitution(P_M, L_M, U_M, b - np.matmul(A, x0))
    q = np.zeros((len(r), m+1))
    h = np.zeros((m+1, m+1))
    q[:, 0] = r/np.linalg.norm(r)
    for k in range(m):
        y = back_substitution(P_M, L_M, U_M, np.matmul(A, q[:, k]))
        for j in range(k):
            h[j, k] = np.matmul(np.transpose(y), q[:, j])
            y -= h[j, k] * q[:, j]
        h[k+1, k] = np.linalg.norm(y)
        if h[k+1, k] != 0:
            q[:, k+1] = y/h[k+1, k]
        b1 = np.zeros(k+2)
        b1[0] = np.linalg.norm(r)
        Q, R = householder(h[:(k+2), :(k+1)])
        P, L, U = pa_lu_factorization(R[:(k+1), :(k+1)])
        c = back_substitution(P, L, U, np.matmul(np.transpose(Q), b1)[:(k+1)])
        x = np.matmul(q[:,:(k+1)], c) + x0
        if h[k+1, k] == 0:
            return x
    return x
