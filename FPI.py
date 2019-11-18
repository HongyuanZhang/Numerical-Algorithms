'''
Fixed Point Iteration Method for solving g(x)=x
'''


#Fixed Point Iteration Method
#Stopping Criteria: when num_step steps have been executed
def FPI(g, x0, num_step):
    x = x0
    #continuously do x_{i+1} <- g(x_{i})
    for i in range(num_step):
        x = g(x)
    return x
