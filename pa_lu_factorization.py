'''
PA=LU factorization
'''
import numpy as np


# this function performs PA=LU factorization of A and returns P, L and U in order
def pa_lu_factorization(A):
    # dimension of the matrix
    n = len(A)
    # placeholder for P
    P = np.identity(n, dtype=float)
    # placeholder for the lower triangular matrix L
    L = np.identity(n, dtype=float)
    # perform elimination from left to right, top to bottom
    for j in range(n-1):
        # exchange rows if necessary
        index = j + np.argmax(abs(A)[j:n,j])
        temp_row = A[j,:].copy()
        A[j,:] = A[index,:]
        A[index,:] = temp_row
        temp_row_P = P[j,:].copy()
        P[j,:] = P[index, :]
        P[index,:] = temp_row_P
        #no singular matrix
        if abs(A[j][j]) < np.finfo(float).eps:
            print("singular matrix encountered")
            return
        # elimination
        for i in range(j+1, n):
            mult = A[i][j]/A[j][j]
            A[i][j] = mult
            for k in range(j+1, n):
                A[i][k] -= mult*A[j][k]
    # fill in the placeholder for L with multipliers
    for i in range(n):
        for j in range(i):
            L[i][j] = A[i][j]
    # since U should be an upper triangular matrix, set lower part of the matrix to 0
    for i in range(n):
        for j in range(i):
            A[i][j] = 0
    return P, L, A


# given the PA=LU factorization of A, and b, solve for x
# prerequisite: A is at least 2*2
def back_substitution(P, L, U, b):
    # dimension of the matrix
    n = len(b)
    # initialize c
    c = np.zeros(n)
    # initialize x
    x = np.zeros(n)
    # precalculate Pb
    Pb = np.matmul(P,b)
    # solve for c from top to bottom
    c[0] = Pb[0]
    for i in range(1, n):
        partial_sum = 0
        for j in range(i):
            partial_sum += L[i][j]*c[j]
        c[i] = Pb[i]-partial_sum
    # solve for x from bottom up
    x[-1] = c[-1]/U[-1][-1]
    for i in range(n-2, -1, -1):
        partial_sum = 0
        for j in range(n-1, i, -1):
            partial_sum += U[i][j]*x[j]
        x[i] = (c[i] - partial_sum)/U[i][i]
    return x
