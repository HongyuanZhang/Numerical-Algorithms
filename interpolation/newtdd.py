'''
Newton's Divided Differences Method for Polynomial Interpolation
'''
import numpy as np
import math
import fundamentals.nest as nest
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


# plot graph of interpolating polynomial
# Input: c, coefficients for the polynomial
#        x_a, left end of the interval being plotted
#        x_b, right end of the interval being plotted
#        x_coords, input points' x-coordinates, used as based points in nest()
def plot_curve(c, x_a, x_b, x_coords):
    # create an array of dense points in [x_a, x_b]
    x = np.linspace(x_a, x_b, (x_b-x_a)*100)
    # degree of polynomial
    degree = len(x_coords)-1
    # evaluate at x
    y = nest.nest(degree, c, x, x_coords)
    # plotting
    plt.figure()
    plt.plot(x, y)
    plt.title('plot of the interpolating polynomial')
    plt.grid(True)
    plt.show()


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


# Interpolating sin(x) by Newton's Divided Differences Method
# Input: x, the point we are interested in getting sin(x)
def sin1(x):
    # input points' x-coordinates
    b = (math.pi/6)*np.arange(4)
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


# Interpolating cos(x) by Newton's Divided Differences Method
# Input: x, the point we are interested in getting cos(x)
def cos1(x):
    # input points' x-coordinates
    b = (math.pi/6)*np.arange(4)
    # input points' y-coordinates
    yb = np.cos(b)
    # get coefficients for the interpolating polynomial
    c = newtdd(b, yb)
    # sign of the result
    s = 1
    # convert x1 into [0, pi/2], the fundamental domain
    x1 = x % (2*math.pi)
    if x1 > math.pi:
        x1 = 2*math.pi-x1
    if x1 > math.pi/2:
        x1 = math.pi - x1
        s = -1
    return s*nest.nest(3, c, x1, b)
