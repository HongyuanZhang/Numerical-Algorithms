'''
Chebyshev Interpolating Polynomial
'''
import numpy as np
import math
import nest
import matplotlib.pyplot as plt

# Newton's Divided Differences
# Input: x, x-coordinates of input points
#        y, y-coordinates of input points
def newtdd(x, y):
    # number of input points
    n = len(x)
    # initialize the triangle of coefficients
    v = np.zeros((n, n))
    # set first column of the triangle to y-coordinates of input points
    v[:,0] = y
    # calculate other parts of triangle, from left to right, top to bottom
    for j in range(1, n):
        for i in range(n-j):
            v[i,j] = (v[i+1,j-1]-v[i,j-1])/(x[i+j]-x[i])
    # return the coefficients for interpolating polynomial
    return v[0,:]


# evaluate the interpolating polynomial at x value(s)
# Input: x, x-coordinates of input points
#        y, y-coordinates of input points
#        x0, x values whose y values will be interpolated using newtdd
def polyinterp(x, y, x0):
    # get interpolating polynomial's coefficients
    c = newtdd(x, y)
    # degree of the interpolating polynomial
    degree = len(x)-1
    # use nest() to evaluate the interpolating polynomial at x0
    return nest.nest(degree, c, x0, x)


# Interpolating sin(x) by Newton's Divided Differences Method using Chebyshev roots
# Input: x, the point we are interested in getting sin(x)
def sin1(x):
    # input points' x-coordinates
    n = 4
    b = (math.pi/4)+(math.pi/4)*np.cos((2*np.arange(1,n+1)-1)*math.pi/(2*n))
    # input points' y-coordinates
    yb = np.sin(b)
    # get coefficients for the interpolating polynomial
    c = newtdd(b, yb)
    # sign of the result
    s = 1
    # convert x1 into [0, pi/2], the fundamental domain
    x1 = x % (2*math.pi)
    if x1 > math.pi:
        x1 = 2*math.pi-x1
        s = -1
    if x1 > math.pi/2:
        x1 = math.pi - x1
    return s*nest.nest(3, c, x1, b)


# Find the Chebyshev Interpolating Polynomial for e^|x|
# Input: n, the desired degree of the resulting Chebyshev Interpolating Polynomial
#        x, the point whose y value will be interpolated
def e_abs_x_cheb(n, x):
    # find Chebyshev roots
    x_base = np.cos((2*np.arange(1,n+2)-1)*math.pi/(2*(n+1)))
    x_abs_base = np.abs(x_base)
    # find y values corresponding to these roots
    y = np.exp(x_abs_base)
    # interpolate the y value at x
    return polyinterp(x_base, y, x)


# Find the Chebyshev Interpolating Polynomial for e^(-x^2)
# Input: n, the desired degree of the resulting Chebyshev Interpolating Polynomial
#        x, the point whose y value will be interpolated
def e_neg_sqr_x_cheb(n, x):
    # find Chebyshev roots
    x_base = np.cos((2*np.arange(1,n+2)-1)*math.pi/(2*(n+1)))
    x_neg_sqr_base = (-1)*np.square(x_base)
    # find y values corresponding to these roots
    y = np.exp(x_neg_sqr_base)
    # interpolate the y value at x
    return polyinterp(x_base, y, x)


# generate plots
n = 20 # desired degree of interpolating polynomial
x = np.linspace(-1,1,201) # points for generating plot
ordinary_x = np.linspace(-1,1,n+1) # the evenly spaced base points
ordinary_y = [np.exp((-1)*np.square(x)) for x in ordinary_x] # y values corresponding to ordinary_x
# Interpolated values for x using the Chebyshev Interpolating Polynomial
cheb_interp_y = [e_neg_sqr_x_cheb(n, item) for item in x]
# Interpolated values for x using the Interpolating Polynomial based on ordinary_x
ord_interp_y = [polyinterp(ordinary_x, ordinary_y, item) for item in x]
# True y values for x
true_y = np.exp((-1)*np.square(x))
# plotting
plt.figure()
plt.plot(x, cheb_interp_y, label = "chebyshev")
plt.plot(x, ord_interp_y, label = "evenly spaced")
plt.legend(loc='upper left')
plt.show()
