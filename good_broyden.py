'''
Good Broyden Method for Solving Nonlinear Systems of Equations
'''
import numpy as np
import pa_lu_factorization


# x0: initial guess
# F: equations being solved
# A0: initial guess for Jacobian matrix of F
# k: number of iterations
def good_broyden(x0, F, A0, k):
    # initialize A, the approximation to Jacobian matrix of F
    A = A0
    # initialize x, solution to be returned
    x = x0
    # initial x_old, used for tracking the previous x value
    x_old = x.copy()
    for i in range(k):
        P, L, U = pa_lu_factorization.pa_lu_factorization(A)
        s = pa_lu_factorization.back_substitution(P, L, U, -F(x))
        # keep a copy of old x
        x_old = x.copy()
        # update x
        x += s
        Delta = F(x)-F(x_old)
        delta = x - x_old
        # update A
        A += np.outer(Delta-np.dot(A, delta), delta)/np.dot(delta, delta)
        print(x)
    return x
