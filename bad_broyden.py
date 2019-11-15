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

# Example Problem: Example 2.33
x0 = np.array([2,2], dtype=float)

def F(x):
    f1 = 6*x[0]**3+x[0]*x[1]-3*x[1]**3-4
    f2 = x[0]**2-18*x[0]*x[1]**2+16*x[1]**3+1
    return np.array([f1, f2])

bad_broyden(x0, F, np.identity(2), 50)