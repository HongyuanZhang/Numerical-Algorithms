'''
Bisection Method for finding roots
'''
import math


# bisection method that takes tolerance as stopping criteria
def bisection_tol(func, a, b, tol):
    # check if f(a)*f(b)<0 (required)
    if func(a)*func(b) >= 0:
        print("[a,b] is not a valid starting interval")
        return math.nan
    else:
        f_a = func(a)
        f_b = func(b)
        # while the interval is wider than 2*tolerance
        while (b-a)/2 > tol:
            # find midpoint
            c = (a+b)/2
            f_c = func(c)
            # if f(c)=0, root is found!
            if f_c == 0:
                return c
            # if f(a)*f(c)<0, pick the half interval [a,c]
            elif f_a*f_c < 0:
                b = c
                f_b = f_c
            # else f(b)*f(c)<0, pick the other half interval [b,c]
            else:
                a = c
                f_a = f_c
        # return the midpoint
        return (a+b)/2


# bisection method that stops when num_step iterations have been executed
def bisection_step(func, a, b, num_step):
    if func(a)*func(b) >= 0:
        print("[a,b] is not a valid starting interval")
        return math.nan
    else:
        f_a = func(a)
        f_b = func(b)
        for i in range(num_step):
            c = (a+b)/2
            f_c = func(c)
            if f_c == 0:
                return c
            elif f_a*f_c < 0:
                b = c
                f_b = f_c
            else:
                a = c
                f_a = f_c
        return (a+b)/2
