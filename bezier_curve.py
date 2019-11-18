'''
Bezier Curve Interpolation
'''
import numpy as np
import matplotlib.pyplot as plt


# Calculate coefficients for the bezier curve for given points
# x, y: x-coordinates and y-coordinates of 4 points (end, control, control, end)
def bezier_curve(x, y):
    bx = 3*(x[1]-x[0])
    cx = 3*(x[2]-x[1])-bx
    dx = x[3]-x[0]-bx-cx
    by = 3*(y[1]-y[0])
    cy = 3*(y[2]-y[1])-by
    dy = y[3]-y[0]-by-cy
    # return coefficients for x(t) and y(t)
    return [bx, cx, dx], [by, cy, dy]


# plot the bezier curve for given points
# x, y: x-coordinates and y-coordinates of 4 points (end, control, control, end)
def plot_bezier_curve(x, y):
    # coefficients for x(t) and y(t)
    coef_x, coef_y = bezier_curve(x, y)
    # t sample points
    t_range = np.linspace(0, 1, 100)
    # corresponding x and y values
    x_val = x[0]+coef_x[0]*t_range+coef_x[1]*np.square(t_range)+coef_x[2]*np.power(t_range, 3)
    y_val = y[0]+coef_y[0]*t_range+coef_y[1]*np.square(t_range)+coef_y[2]*np.power(t_range, 3)
    # plot
    plt.plot(x_val, y_val)
    plt.show()
