# Begin your comment with a pound sign
# To define a function, first type the keyword ``def'' followed by function name
# Then write input arguments inside a pair of parenthesis and then a colon
# In this case, there is only one input argument, n, for which we want to calculate
# \sum_{i=1}^n i^2
def sum_of_squares(n):
    # main body of the function
    # initialize current sum to 0
    sum = 0
    # to write a for-loop in Python, use the format ``for x in [x_1, x_2, ...]''
    # range() creates a range object for easy iteration
    # for more information of how to use range(), check its documentation
    # at https://docs.python.org/3/library/functions.html#func-range
    # Often, documentation is your best friend when you start to use a new function
    for i in range(1, n+1):
        # this line updates the sum and is equivalent to
        # sum = sum + i**2
        # ``y ** z'' is one way to raise powers in Python, meaning ``y^z''
        sum += i**2
    # return the updated sum as output
    return sum
