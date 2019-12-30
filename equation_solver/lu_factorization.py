'''
LU factorization
'''
import numpy as np


# this function performs LU factorization of A and returns L and U in order
def lu_factorization(A):
    # dimension of the matrix
    n = len(A)
    # placeholder for the lower triangular matrix L
    L = np.identity(n)
    # perform elimination from left to right, top to bottom
    for j in range(n-1):
        # no zero pivot
        if abs(A[j][j]) < np.finfo(float).eps:
            print("zero pivot encountered")
            return
        for i in range(j+1, n):
            mult = A[i][j]/A[j][j]
            L[i][j] = mult
            for k in range(j+1, n):
                A[i][k] -= mult*A[j][k]
    # since U should be an upper triangular matrix, set lower part of the matrix to 0
    for i in range(n):
        for j in range(i):
            A[i][j] = 0
    return L, np.array(A)


# given the LU factorization of A, and b, solve for x
# prerequisite: A is at least 2*2
def back_substitution(L, U, b):
    # dimension of the matrix
    n = len(b)
    # initialize c
    c = np.zeros(n)
    # initialize x
    x = np.zeros(n)
    # solve for c from top to bottom
    c[0] = b[0]
    for i in range(1, n):
        partial_sum = 0
        for j in range(i):
            partial_sum += L[i][j]*c[j]
        c[i] = b[i]-partial_sum
    # solve for x from bottom up
    x[-1] = c[-1]/U[-1][-1]
    for i in range(n-2, -1, -1):
        partial_sum = 0
        for j in range(n-1, i, -1):
            partial_sum += U[i][j]*x[j]
        x[i] = (c[i] - partial_sum)/U[i][i]
    return x
