'''
Numerical Integration using Newton-Cotes Formulas
'''
import numpy as np


# Integration by Composite Trapezoid Rule, a closed newton-cotes method
# f: integrand
# a: lower end of the integral
# b: upper end of the integral
# m: number of panels
def int_trapezoid(f, a, b, m):
    h = (b-a)/m
    panel_x_points = np.linspace(a, b, m+1)
    panel_y_points = f(panel_x_points)
    return (h/2)*(panel_y_points[0]+ panel_y_points[-1]+2*np.sum(panel_y_points[1:m]))


# Integration by Composite Simpson's Rule, a closed newton-cotes method
# f: integrand
# a: lower end of the integral
# b: upper end of the integral
# m: number of panels
def int_simpson(f, a, b, m):
    h = (b-a)/(2*m)
    panel_x_points = np.linspace(a, b, 2*m + 1)
    panel_y_points = f(panel_x_points)
    first_sum = 0
    for i in range(1, m+1):
        first_sum += panel_y_points[2*i-1]
    second_sum = 0
    for i in range(1, m):
        second_sum += panel_y_points[2*i]
    return (h/3)*(panel_y_points[0]+panel_y_points[-1]+4*first_sum+2*second_sum)


# Integration by Composite Midpoint Rule, an open newton-cotes method
# f: integrand
# a: lower end of the integral
# b: upper end of the integral
# m: number of panels
def int_midpoint(f, a, b, m):
    h = (b-a)/m
    panel_x_points = np.linspace(a, b, m+1)
    mid_pts = []
    for i in range(m):
        mid_pts.append((panel_x_points[i]+panel_x_points[i+1])/2)
    mid_pts_f_vals = f(mid_pts)
    return h*np.sum(mid_pts_f_vals)
