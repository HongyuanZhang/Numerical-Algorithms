'''
The Method of False Position for solving equations
'''
import math


# method of false position that stops when num_step iterations have been executed
def false_position(func, a, b, num_step):
    if func(a)*func(b) >= 0:
        print("[a,b] is not a valid starting interval")
        return math.nan
    else:
        f_a = func(a)
        f_b = func(b)
        for i in range(num_step):
            c = (b*f_a-a*f_b)/(f_a-f_b)
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
