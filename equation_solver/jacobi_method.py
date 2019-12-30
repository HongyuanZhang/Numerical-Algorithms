'''
Jacobi Method - Iterative Method for Solving Systems of Equations
'''
import numpy as np


# Jacobi Method for Solving Systems of Equations
# Input: a, strictly diagonally dominant coefficient matrix
#        b, right-hand-side of the system of equations
#        x0, initial guess for solution
#        k, desired number of iterations
def jacobi(a, b, x0, k):
    # diagonal matrix of a
    d = np.diag(np.diag(a))
    # r = L + U
    r = a - d
    x = x0
    for i in range(k):
        # Jacobi Iteration
        x = np.linalg.inv(d).dot(b-r.dot(x))
    return x
