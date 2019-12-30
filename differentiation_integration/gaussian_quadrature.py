'''
Gaussian Quadrature
'''
import math
import numpy as np


# Numerical Integration by Gaussian Quadrature
# f: integrand
# a: lower end of the integral
# b: upper end of the integral
# n: number of degrees of the Legendre polynomial used (possible values: 2, 3, 4)
def int_gq(f, a, b, n=4):
    roots = [[-math.sqrt(1/3), math.sqrt(1/3)],
             [-math.sqrt(3/5), 0, math.sqrt(3/5)],
             [-math.sqrt((15+2*math.sqrt(30))/35), -math.sqrt((15-2*math.sqrt(30))/35),
              math.sqrt((15-2*math.sqrt(30))/35), math.sqrt((15+2*math.sqrt(30))/35)]]
    coefs = [[1,1],
             [5/9, 8/9, 5/9],
             [(90-5*math.sqrt(30))/180, (90+5*math.sqrt(30))/180,
              (90+5*math.sqrt(30))/180, (90-5*math.sqrt(30))/180]]
    root_n = np.asarray(roots[n-2])
    input_n = ((b-a)*root_n+b+a)/2
    coef_n = np.asarray(coefs[n-2])
    return np.sum(coef_n*f(input_n))*(b-a)/2
