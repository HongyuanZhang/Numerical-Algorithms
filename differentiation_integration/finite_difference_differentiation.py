'''
Numerical Differentiation by Finite Difference
(First & Second Derivatives)
'''


# first order method for approximating the first derivative
def two_point_forward_difference(f, x, h):
    return (f(x+h)-f(x))/h


# second order method for approximating the first derivative
def three_pt_centered_difference(f, x, h):
    return (f(x+h)-f(x-h))/(2*h)


# second order method for approximating the second derivative
def three_pt_centered_difference_sec(f, x, h):
    return (f(x-h)-2*f(x)+f(x+h))/h**2
