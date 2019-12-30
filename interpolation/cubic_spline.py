'''
Cubic Splines for Interpolation
'''
import numpy as np
import equation_solver.pa_lu_factorization as pa_lu_factorization
import matplotlib.pyplot as plt


# Give cubic spline coefficients for interpolating points
# whose x-coordinates are in x and y-coordinates are in y
# and whose end condition is specified by end_condition
def spline_coeff(x, y, end_condition, v1=0, vn=0):
    n = len(x)  # number of points
    A = np.zeros((n, n))
    r = np.zeros((n, 1))
    dx = np.zeros(n-1)
    dy = np.zeros(n-1)
    # calculate deltas and Deltas
    for i in range(n-1):
        dx[i] = x[i+1]-x[i]
        dy[i] = y[i+1]-y[i]
    # Fill in the matrix A and the right-hand side r
    for i in range(1, n-1):
        A[i, i-1] = dx[i-1]
        A[i, i] = 2 * (dx[i-1]+dx[i])
        A[i, i+1] = dx[i]
        r[i] = 3 * (dy[i]/dx[i]-dy[i-1]/dx[i-1])
    # natural spline conditions
    if end_condition == "natural":
        A[0,0] = 1
        A[n-1, n-1] = 1
    # curvature-adj conditions
    elif end_condition == "curvature-adjusted":
        A[0,0] = 2
        r[0] = v1
        A[n-1, n-1] = 2
        r[n-1] = vn
    # clamped conditions
    elif end_condition == "clamped":
        A[0, 0] = 2 * dx[0]
        A[0, 1] = dx[0]
        r[0] = 3 * (dy[0]/dx[0]-v1)
        A[n-1, n-2] = dx[n-2]
        A[n-1, n-1] = 2 * dx[n-2]
        r[n-1] = 3 * (vn-dy[n-2]/dx[n-2])
    # parabolic conditions
    elif end_condition == "parabolic":
        A[0, 0] = 1
        A[0, 1] = -1
        A[n-1, n-2] = 1
        A[n-1, n-1] = -1
    # not-a-knot conditions
    elif end_condition == "not-a-knot":
        A[0, 0] = dx[1]
        A[0, 1] = -(dx[0]+dx[1])
        A[0, 2] = dx[0]
        A[n-1, n-3] = dx[n-2]
        A[n-1, n-2] = -(dx[n-3]+dx[n-2])
        A[n-1, n-1] = dx[n-3]
    coeff = np.zeros((n, 3))
    # solve for c coefficients
    P, L, U = pa_lu_factorization.pa_lu_factorization(A)
    coeff[:, 1] = pa_lu_factorization.back_substitution(P, L, U, r)
    # solve for b and d
    for i in range(n-1):
        coeff[i, 2] = (coeff[i+1, 1] - coeff[i, 1]) / (3 * dx[i])
        coeff[i, 0] = dy[i]/dx[i] - dx[i]*(2*coeff[i,1]+coeff[i+1,1])/3
    # cubic spline coefficients
    return coeff[0:n-1, 0:3]


# plot the cubic spline returned by spline_coeff
# x, y, end_condition denote the same input as in the above function
# k: number of points between x_i's used for plotting
def spline_plot(x, y, k, end_condition, v1=0, vn=0):
    n = len(x)  # number of points
    coeff = spline_coeff(x, y, end_condition, v1, vn)  # cubic spline coefficients
    x1 = []
    y1 = []
    # loop through points
    for i in range(n-1):
        xs = np.linspace(x[i], x[i+1], k+1)
        dx = xs - x[i]
        # evaluating using nested multiplication
        ys = coeff[i, 2] * dx
        ys = (ys + coeff[i, 1]) * dx
        ys = (ys + coeff[i, 0]) * dx + y[i]
        x1.extend(xs[0:k])
        y1.extend(ys[0:k])
    x1.append(x[-1])
    y1.append(y[-1])

    # Plot the data
    fig, ax = plt.subplots(1, figsize=(8, 6))
    fig.suptitle('Cubic Spline')

    plt.scatter(x1, y1)

    # Show the grid lines as dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')

    plt.show()
