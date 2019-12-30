'''
Gauss-Seidel Method - Iterative Method for Solving Systems of Equations
'''
import numpy as np


# Gauss-Seidel Method for Solving Systems of Equations
# Input: a, strictly diagonally dominant coefficient matrix
#        b, right-hand-side of the system of equations
#        x0, initial guess for solution
#        k, desired number of iterations
def gauss_seidel(a, b, x0, k):
    # diagonal matrix of a
    d = np.diag(np.diag(a))
    # dimension
    n = len(x0)
    # lower triangle of a (entries below the main diagonal)
    L = np.tril(a) - d
    # upper triangle of a (entries above the main diagonal)
    U = np.triu(a) - d
    x = x0
    for i in range(k):
        # Gauss-Seidel Iteration
        # save the previous x, i.e. x_k
        xk = x
        for j in range(n):
            # calculate x_{k+1}
            x = np.linalg.inv(d).dot(b-U.dot(xk)-L.dot(x))
    return x
