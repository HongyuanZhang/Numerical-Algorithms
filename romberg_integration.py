'''
Numerical Integration: Romberg Integration
'''
import numpy as np


# Romberg Integration
# f: integrand
# a: lower end of the integral
# b: upper end of the integral
# tol: tolerance used to stop the algorithm
def int_romberg(f, a, b, tol):
    cur_row = [(b-a)*(f(a)+f(b))/2]
    j = 2
    while True:
        prev_row = cur_row.copy()
        h = (b-a)/2**(j-1)
        cur_row = np.zeros(j)
        summand = [f(a+(2*i-1)*h) for i in range(1, 2**(j-2)+1)]
        cur_row[0] = 0.5*prev_row[0]+h*np.sum(summand)
        for k in range(1, j):
            cur_row[k] = (4**k*cur_row[k-1]-prev_row[k-1])/(4**k-1)
        if abs(cur_row[-1]-prev_row[-1]) < tol:
            break
        j += 1
    return cur_row[-1]
