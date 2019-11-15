'''
Nested Multiplication
'''


#To evaluate the following function at multiple x's, input an numpy array of x's
def nest(degree, coef, x, base_pt):
    y = coef[degree]
    for i in range(degree-1, -1, -1):
        y = y*(x-base_pt[i])+coef[i]
    return y
