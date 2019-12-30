"""
Finite Difference Method for Solving Hyperbolic PDEs
of the form u_{tt} + 2*alpha*u_t + beta^2 * u = u_{xx} + f(x,t),
alpha > beta >= 0
defined on a rectangular domain with Dirichlet boundaries
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# conditionally stable finite-difference scheme for solving
# the wave equation introduced in Sauer
def wave_eq_cond_stable(xl, xr, yb, yt, M, N, f, g, l, r, C):
    '''
    :param xl: lower limit of the space variable
    :param xr: upper limit of the space variable
    :param yb: beginning time
    :param yt: terminal time
    :param M: number of space steps
    :param N: number of time steps
    :param f: initial condition
    :param g: u_t at (x,0)
    :param l: lower end Dirichlet boundary condition
    :param r: right end Dirichlet boundary condition
    :param C: as in the wave eq: u_{tt} = C^2 * u_{xx}
    :return: None. Plot the approximated solution.
    '''
    h = (xr - xl) / M  # space step
    k = (yt - yb) / N  # time step
    sigma = C * k / h
    m = M - 1
    n = N
    a = np.zeros((m, m))
    np.fill_diagonal(a, 2 - 2 * sigma**2)
    rng = np.arange(m - 1)
    a[rng, rng + 1] = sigma**2
    a[rng + 1, rng] = sigma**2
    lside = l(yb + np.arange(n + 1) * k)  # solution values on the left side
    rside = r(yb + np.arange(n + 1) * k)  # solution values on the right side
    x_grid = xl+np.arange(1, M) * h
    w = np.zeros((m, N + 1))  # grid
    w[:, 0] = f(xl + (np.arange(m) + 1) * h)  # initial condition
    vec = np.zeros(m)
    vec[0] = l(yb)
    vec[-1] = r(yb)
    w[:, 1] = 0.5 * np.matmul(a, f(x_grid)) + k * g(x_grid) + 0.5 * sigma**2 * vec
    # fill in interior points on the grid
    for j in range(1, n):
        vec[0] = l(yb + j * k)
        vec[-1] = r(yb + j * k)
        w[:, j + 1] = np.matmul(a, w[:, j]) - w[:, j-1] + sigma**2 * vec
    # compile the whole solution
    w = np.vstack((lside, w, rside))
    # grid for plotting
    x = xl + np.arange(M + 1) * h
    t = yb + np.arange(N + 1) * k
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


# unconditionally stable finite-difference scheme for solving general
# hyperbolic PDEs, introduced in the paper by Mohanty
# citation: https://www.sciencedirect.com/science/article/pii/S0893965904900195
def hyperbolic_pde_mohanty(alpha, beta, f, yt, phi, phixx, psi, g0, g1, h, k):
    '''
    :param alpha: see comments at the top
    :param beta: see comments at the top
    :param f: see comments at the top
    :param yt: terminal time
    :param phi: initial condition
    :param phixx: phi_{xx} at (x,0)
    :param psi: u_t at (x,0)
    :param g0: lower end Dirichlet boundary condition
    :param g1: right end Dirichlet boundary condition
    :param h: space step size
    :param k: time step size
    :return: None. Plot the approximated solution.
    '''
    a = alpha*k
    b = beta**2 * k**2
    l = k/h
    sigma = 1/64
    gamma = 1/4
    m = int(1/h)-1
    n = int(yt/k)
    u = np.zeros((m+2, n+1))  # grid
    row = np.arange(m+2)*h
    col = np.arange(n+1)*k
    u[:, 0] = phi(row)  # initial condition
    u[0, :] = g0(col)  # solution values on the left side
    u[-1, :] = g1(col)  # solution values on the right side
    x_grid = (np.arange(m)+1)*h
    # second-order approximation to u at t=k
    utt0x = phixx(x_grid) - 2*alpha*psi(x_grid) - beta**2 * phi(x_grid) + f(x_grid, 0)
    u[1:-1, 1] = u[1:-1, 0] + k*psi(x_grid) + (k**2/2)*utt0x
    A = np.zeros((m, m))
    np.fill_diagonal(A, 1+sigma*b**2+2*gamma*l**2+a)
    rng = np.arange(m - 1)
    A[rng, rng + 1] = -gamma*l**2
    A[rng + 1, rng] = -gamma*l**2
    B = np.zeros((m, m))
    np.fill_diagonal(B, 2*l**2+b-2-2*sigma*b**2-4*gamma*l**2)
    B[rng, rng + 1] = 2*gamma*l**2 - l**2
    B[rng + 1, rng] = 2*gamma*l**2 - l**2
    C = np.zeros((m, m))
    np.fill_diagonal(C, 1+sigma*b**2+2*gamma*l**2-a)
    C[rng, rng + 1] = -gamma*l**2
    C[rng + 1, rng] = -gamma*l**2
    # fill in interior points on the grid
    for j in range(2, n+1):
        row_f = f(x_grid, j*k)
        u[1:-1, j] = np.matmul(np.linalg.inv(A), k**2*row_f-np.matmul(B,u[1:-1,j-1])-np.matmul(C, u[1:-1, j-2]))
    # grid for plotting
    x = np.arange(m+2) * h
    t = np.arange(n+1) * k
    xv, tv = np.meshgrid(x, t)
    # plot approximated solution
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xv, tv, u.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
'''
Example Usage: (Example 8.6 in 2nd Edition Sauer)
alpha = 0
beta = 0
def f(x, t):
    return np.zeros(len(x))
yt = 1
def phi(x):
    return np.sin(math.pi*x)
def phixx(x):
    return -math.pi**2*np.sin(math.pi*x)
def psi(x):
    return np.zeros(len(x))
def g0(x):
    return np.zeros(len(x))
def g1(x):
    if isinstance(x, (int, float)):
        return 0
    else:
        return np.zeros(len(x))
h=1.0/20
k=1.0/30
hyperbolic_pde_mohanty(alpha, beta, f, yt, phi, phixx, psi, g0, g0, h, k)
wave_eq_cond_stable(0, 1, 0, yt, 20, 30, phi, psi, g1, g1, 2)
'''