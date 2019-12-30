'''
Newton's Method for Solving Nonlinear Systems of Equations
'''
import pa_lu_factorization


# x0: initial guess
# F: equations being solved
# DF: Jacobian matrix of F
# k: number of iterations
def multivar_newton(x0, F, DF, k):
    # initialize x, solution to be returned
    x = x0
    for i in range(k):
        P, L, U = pa_lu_factorization.pa_lu_factorization(DF(x))
        s = pa_lu_factorization.back_substitution(P, L, U, -F(x))
        # update x
        x += s
    return x
