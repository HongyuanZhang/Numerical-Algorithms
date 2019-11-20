'''
Adaptive Quadrature for Numerical Integration
'''


# Adaptive Quadrature
# f: integrand
# a: lower end of the integral
# b: upper end of the integral
# tol: tolerance used to stop the algorithm
def int_aq(f, a, b, tol):
    return int_aq_helper(f, a, b, b-a, tol)


# recursive helper
# diff: length of original integration interval
def int_aq_helper(f, a, b, diff, tol):
    c = (a+b)/2
    s_ab = trap(f, a, b)
    s_ac = trap(f, a, c)
    s_cb = trap(f, c, b)
    if abs(s_ab-s_ac-s_cb) < 3*tol*(b-a)/diff:
        return s_ac+s_cb
    else:
        return int_aq_helper(f, a, c, diff, tol) + int_aq_helper(f, c, b, diff, tol)


# Trapezoid Rule
# Note: This can be replaced by Simpson's Rule / Midpoint Rule / Other Rules...
def trap(f, a, b):
    return (b-a)*(f(a)+f(b))/2
