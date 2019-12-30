'''
Nested Multiplication
'''


# note: coef and base_pt are numpy arrays
# x can be a single number, if you want to evaluate at a single point
# or a numpy array, if you want to evaluate at multiple points
def nest(degree, coef, x, base_pt):
    y = coef[degree]
    for i in range(degree-1, -1, -1):
        y = y*(x-base_pt[i])+coef[i]
    return y
