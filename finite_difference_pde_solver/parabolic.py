"""
Crank-Nicolson Method for Solving Parabolic PDEs
of the form: u_t=Du_{xx}+Cu (reaction diffusion equation)
defined on a rectangular domain with Dirichlet boundaries
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def parabolic_pde_cn(xl, xr, yb, yt, M, N, f, l, r, D, C):
    '''
    :param xl: lower limit of the space variable
    :param xr: upper limit of the space variable
    :param yb: beginning time
    :param yt: terminal time
    :param M: number of space steps
    :param N: number of time steps
    :param f: initial condition
    :param l: lower end Dirichlet boundary condition
    :param r: right end Dirichlet boundary condition
    :param D: see comments at the top
    :param C: see comments at the top
    :return: None. Plot the approximated solution.
    '''
    h = (xr-xl)/M  # space step
    k = (yt-yb)/N  # time step
    sigma = D*k/(h*h)
    m = M-1
    n = N
    a = np.zeros((m, m))
    np.fill_diagonal(a, 2+2*sigma-k*C)
    rng = np.arange(m-1)
    a[rng, rng+1] = -sigma
    a[rng+1, rng] = -sigma
    b = np.zeros((m, m))
    np.fill_diagonal(b, 2-2*sigma+k*C)
    b[rng, rng + 1] = sigma
    b[rng + 1, rng] = sigma
    lside = l(yb+np.arange(n+1)*k)  # solution values on the left side
    rside = r(yb+np.arange(n+1)*k)  # solution values on the right side
    w = np.zeros((m, N+1))  # grid
    w[:, 0] = f(xl+(np.arange(m)+1)*h)  # initial condition
    # fill in interior points on the grid
    for j in range(n):
        sides = np.zeros(m)
        sides[0] = lside[j]+lside[j+1]
        sides[-1] = rside[j]+rside[j+1]
        w[:, j+1] = np.linalg.solve(a, np.matmul(b, w[:, j])+sigma*sides)
    # compile the whole solution
    w = np.vstack((lside, w, rside))
    # grid for plotting
    x = xl + np.arange(M+1)*h
    t = yb + np.arange(N+1)*k
    xv, tv = np.meshgrid(x, t)
    # plot approximated solution
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xv, tv, w.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
'''
Example Usage: (Example 8.5 in 2nd Edition Sauer)
D=1
C=9.5
def f(x):
    return np.square(np.sin(math.pi*x))
def l(t):
    return np.zeros(len(t))
def r(t):
    return l(t)
parabolic_pde_cn(0,1,0,2,20,40,f,l,r,D,C)
'''