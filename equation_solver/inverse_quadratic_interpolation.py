'''
Inverse Quadratic Interpolation Method for solving equations
'''


# Inverse Quadratic Interpolation Method
# Stopping Criteria: when the approximated root converges
def iqi(g, x0, x1, x2, epsilon):
    cur_x=x2
    prev_x=x1
    prev_prev_x=x0
    prev_prev_prev_x=x0
    # continuously do iqi method until cur_x and prev_x are close
    while abs(cur_x - prev_x) > epsilon:
        prev_prev_prev_x = prev_prev_x
        prev_prev_x = prev_x
        prev_x = cur_x
        q=g(prev_prev_prev_x)/g(prev_prev_x)
        r=g(prev_x)/g(prev_prev_x)
        s=g(prev_x)/g(prev_prev_prev_x)
        cur_x = prev_x - (r*(r-q)*(prev_x-prev_prev_x)+(1-r)*s*(prev_x-prev_prev_prev_x))/((q-1)*(r-1)*(s-1))
    return cur_x
