'''
Bad Broyden Method for Solving Nonlinear Systems of Equations
'''
import numpy as np


# x0: initial guess
# F: equations being solved
# B0: initial guess for inverse of Jacobian matrix of F
# k: number of iterations
def bad_broyden(x0, F, B0, k):
    # initialize B, the approximation to inverse of Jacobian matrix of F
    B = B0
    # initialize x, solution to be returned
    x = x0
    # initial x_old, used for tracking the previous x value
    x_old = x.copy()
    for i in range(k):
        # keep a copy of old x
        x_old = x.copy()
        # update x
        x -= np.matmul(B, F(x))
        Delta = F(x)-F(x_old)
        delta = x - x_old
        # update B
        B += np.matmul(np.outer(delta-np.dot(B, Delta), delta), B)/np.dot(np.dot(delta, B), Delta)
        print(x)
    return x
