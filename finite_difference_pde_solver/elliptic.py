"""
Finite Difference Method for Solving Poisson Equations
defined on a rectangular domain with Dirichlet boundaries
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def elliptic_pde(xl, xr, yb, yt, M, N, f, g1, g2, g3, g4):
    '''
    :param xl: lower limit of the first space variable
    :param xr: upper limit of the first space variable
    :param yb: lower limit of the second space variable
    :param yt: upper limit of the second space variable
    :param M: number of space steps for the first space variable
    :param N: number of space steps for the second space variable
    :param f: external forcing function as in the Poisson's eq: Laplacian(u) = f
    :param g1: bottom side Dirichlet boundary condition
    :param g2: top side Dirichlet boundary condition
    :param g3: left side Dirichlet boundary condition
    :param g4: right side Dirichlet boundary condition
    :return: None. Plot the approximated solution.
    '''
    m = M + 1
    n = N + 1
    mn = m * n
    h = (xr-xl)/M  # space step for space variable x
    h2 = h**2
    k = (yt-yb)/N  # space step for space variable y
    k2 = k**2
    # grid
    x = xl + np.arange(M + 1) * h
    y = yb + np.arange(N + 1) * k
    A = np.zeros((mn, mn))
    b = np.zeros((mn, 1))
    # fill in interior points on the grid
    for i in range(1, m-1):
        for j in range(1, n-1):
            A[i+(j-1)*m, i-1+(j-1)*m] = 1/h2
            A[i+(j-1)*m, i+1+(j-1)*m] = 1/h2
            A[i+(j-1)*m, i+(j-1)*m] = -2/h2-2/k2
            A[i+(j-1)*m, i+(j-2)*m] = 1/k2
            A[i+(j-1)*m, i+j*m] = 1/k2
            b[i+(j-1)*m] = f(x[i], y[j])
    #  bottom and top boundary values
    for i in range(m):
        j = 0
        A[i+(j-1)*m, i+(j-1)*m] = 1
        b[i+(j-1)*m] = g1(x[i])
        j = n - 1
        A[i + (j - 1) * m, i + (j - 1) * m] = 1
        b[i + (j - 1) * m] = g2(x[i])
    # left and right boundary values
    for j in range(1, n-1):
        i = 0
        A[i + (j - 1) * m, i + (j - 1) * m] = 1
        b[i + (j - 1) * m] = g3(y[j])
        i = m - 1
        A[i + (j - 1) * m, i + (j - 1) * m] = 1
        b[i + (j - 1) * m] = g4(y[j])
    v = np.linalg.solve(A, b)
    w = v.reshape((m, n))
    w = np.roll(w, 1, axis=0)
    # grid for plotting
    xv, yv = np.meshgrid(x, y)
    # plot approximated solution
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xv, yv, w, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
'''
Example Usage: (Example 8.9 in 2nd Edition Sauer)
def f(x, y):
    return 0
def g1(x):
    return math.sin(math.pi*x)
def g3(x):
    return 0

elliptic_pde(0,1,0,1,10,10,f,g1,g1,g3,g3)
'''