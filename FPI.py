'''
Fixed Point Iteration Method for solving g(x)=x
'''
import math

#example function g(x)
def g(x):
    return math.cos(x)**2

#Fixed Point Iteration Method
#Stopping Criteria: when num_step steps have been executed
def FPI(g, x0, num_step):
    x = x0
    #continuously do x_{i+1} <- g(x_{i})
    for i in range(num_step):
        x = g(x)
    return x

print(FPI(g, 1, 500))