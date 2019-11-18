'''
Successive Over-Relaxation (SOR) Method - Iterative Method for Solving Systems of Equations
'''
import numpy as np


# SOR Method for Solving Systems of Equations
# Input: a, strictly diagonally dominant coefficient matrix
#        b, right-hand-side of the system of equations
#        x0, initial guess for solution
#        k, desired number of iterations
#        w, relaxation parameter
def sor(a, b, x0, k, w):
    # diagonal matrix of a
    d = np.diag(np.diag(a))
    # lower triangle of a (entries below the main diagonal)
    L = np.tril(a) - d
    # upper triangle of a (entries above the main diagonal)
    U = np.triu(a) - d
    x = x0
    for i in range(k):
        # SOR Iteration
        x = np.linalg.inv(w*L+d).dot((1-w)*d.dot(x)-w*U.dot(x))+w*np.linalg.inv(d+w*L).dot(b)
    return x
