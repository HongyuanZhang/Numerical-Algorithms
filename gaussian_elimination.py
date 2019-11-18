'''
Gaussian Elimination and Backward Substitution
'''
import numpy as np


# naive gaussian elimination of the tableau [coef_matrix | b], where row swapping is not allowed
def gaussian_elimination(coef_matrix, b):
    # dimension of the matrix
    n = len(b)
    # perform elimination from left to right, top to bottom
    for j in range(n-1):
        # no zero pivot
        if abs(coef_matrix[j][j]) < np.finfo(float).eps:
            print("zero pivot encountered")
            return
        for i in range(j+1, n):
            mult = coef_matrix[i][j]/coef_matrix[j][j]
            for k in range(j+1, n):
                coef_matrix[i][k] -= mult*coef_matrix[j][k]
            b[i] -= mult*b[j]
    return coef_matrix, b


# given gaussian-eliminated coef_matrix and b, this function solves for x
def back_substitution(reduced_matrix, reduced_b):
    # dimension of the matrix
    n = len(reduced_b)
    # initialize x
    x = np.zeros(n)
    # solve for x from bottom up
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            reduced_b[i] -= reduced_matrix[i][j]*x[j]
        x[i] = reduced_b[i]/reduced_matrix[i][i]
    return x
