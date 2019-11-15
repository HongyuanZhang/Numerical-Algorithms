'''
Cholesky Factorization
'''
import numpy as np
import math

# Find the Cholesky Factorization of A, a symmetric and positive-definite matrix
# Prerequisite: A is a numpy array of floating numbers
def cholesky_factorization(A):
    n = len(A) # dimension
    R = np.zeros((n, n)) # placeholder for R, the upper-triangular matrix with A=R^TR
    # iterate through submatrices
    for i in range(n):
        if A[i, i] < 0:
            return
        R[i, i] = math.sqrt(A[i, i])
        u = 1/R[i, i]*A[i, i+1:n]
        R[i, i+1:n] = u
        A[i+1:n, i+1:n] = np.subtract(A[i+1:n, i+1:n], np.outer(np.transpose(u), u))
    return R

# Example 2.28 on the textbook (Sauer 2nd edition)
A = np.array([[4,-2,2],[-2,2,-4],[2,-4,11]], dtype=float)
print(cholesky_factorization(A))