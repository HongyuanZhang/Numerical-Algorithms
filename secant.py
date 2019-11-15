'''
Secant Method for solving equations
'''


#Secant Method
#Stopping Criteria: when the approximted root converges
def secant(g, x0, x1, epsilon):
    cur_x = x1
    prev_x = x0
    #continuously do x_{i+1} <- x_i-g(x_i)(x_i-x_{i-1})/(g(x_i)-g(x_{i-1}))
    #until cur_x and prev_x are close
    while (abs(cur_x-prev_x) > epsilon):
        prev_prev_x = prev_x
        prev_x = cur_x
        cur_x = prev_x - g(prev_x)*(prev_x-prev_prev_x)/(g(prev_x)-g(prev_prev_x))
    return cur_x
