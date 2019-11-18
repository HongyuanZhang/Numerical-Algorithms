'''
Newton's Method for solving equations
'''
import numpy as np


# Newton's Method
# Stopping Criteria: when the approximated root converges
def Newton(g, g_derivative, x0, epsilon):
    cur_x = x0
    prev_x = x0+epsilon+1
    # continuously do x_{i+1} <- x_i-g(x_i)/g'(x_i) until cur_x and prev_x are close
    while abs(cur_x-prev_x) > epsilon:
        prev_x = cur_x
        cur_x = prev_x-g(prev_x)/g_derivative(prev_x)
    return cur_x
