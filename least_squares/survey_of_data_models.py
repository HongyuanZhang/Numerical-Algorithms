import numpy as np
import math


# periodic model
t=np.array([-2,0,1,2])
y=np.array([1,2,2,5])
A1=np.ones(4)
A2=np.cos(2*math.pi*t)
A3=np.sin(2*math.pi*t)
A=np.column_stack((A1, A2, A3))
ATA=np.matmul(np.transpose(A), A)
ATb=np.matmul(np.transpose(A), y)
x=np.linalg.solve(ATA, ATb)
def f(t):
    return x[0]+x[1]*np.cos(2*math.pi*t)+x[2]*np.sin(2*math.pi*t)
two_norm_error = np.linalg.norm(y-f(t))
SE = np.sum(np.square(y-f(t)))
RMSE = math.sqrt(SE/4)


# exponential model
t=np.array([0,1,1,2])
y=np.array([1,1,2,4])
A1=np.ones(4)
A2=t
A=np.column_stack((A1, A2))
ATA=np.matmul(np.transpose(A), A)
ATb=np.matmul(np.transpose(A), np.log(y))
x=np.linalg.solve(ATA, ATb)
def f(t):
    c1 = math.exp(x[0])
    return c1*np.exp(x[1]*t)
two_norm_error = np.linalg.norm(y-f(t))
SE = np.sum(np.square(y-f(t)))
RMSE = math.sqrt(SE/4)


# power law
t=np.array([1,1,2,3,5])
y=np.array([2,4,5,6,10])
A1=np.ones(5)
A2=np.log(t)
A=np.column_stack((A1, A2))
ATA=np.matmul(np.transpose(A), A)
ATb=np.matmul(np.transpose(A), np.log(y))
x=np.linalg.solve(ATA, ATb)
def f(t):
    c1 = math.exp(x[0])
    return c1*np.power(t, x[1])
two_norm_error = np.linalg.norm(y-f(t))
SE = np.sum(np.square(y-f(t)))
RMSE = math.sqrt(SE/5)


# drug concentration model
t=np.array([1,2,3,4])
y=np.array([2,4,3,2])
A1=np.ones(4)
A2=t
A=np.column_stack((A1, A2))
ATA=np.matmul(np.transpose(A), A)
ATb=np.matmul(np.transpose(A), np.log(y)-np.log(t))
x=np.linalg.solve(ATA, ATb)
def f(t):
    c1 = math.exp(x[0])
    return c1*t*np.exp(x[1]*t)
two_norm_error = np.linalg.norm(y-f(t))
SE = np.sum(np.square(y-f(t)))
RMSE = math.sqrt(SE/4)
