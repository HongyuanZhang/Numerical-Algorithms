import numpy as np


# classical Gram-Schmidt orthogonalization
# A is a matrix (float numpy array) whose columns are linearly independent
def classical_gs(A):
    dim = A.shape
    Q = np.zeros(dim)  # initialize Q
    R = np.zeros((dim[1], dim[1]))  # initialize R
    for j in range(dim[1]):
        y = np.copy(A[:, j])
        for i in range(j):
            R[i, j] = np.matmul(np.transpose(Q[:, i]), A[:, j])
            y -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(y)
        Q[:, j] = y / R[j, j]
    return Q, R


# modified Gram-Schmidt orthogonalization
# A is a matrix (numpy array) whose columns are linearly independent
def modified_gs(A):
    dim = A.shape
    Q = np.zeros(dim)  # initialize Q
    R = np.zeros((dim[1], dim[1]))  # initialize R
    for j in range(dim[1]):
        y = np.copy(A[:, j])
        for i in range(j):
            R[i, j] = np.matmul(np.transpose(Q[:, i]), y)
            y -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(y)
        Q[:, j] = y / R[j, j]
    return Q, R


# use householder reflectors to perform QR factorization of A
# A is a matrix (float numpy array) whose columns are linearly independent
def householder(A):
    dim = A.shape
    householders = []  # record the householder reflectors
    for j in range(dim[1]):
        x = np.copy(A[j:, j])
        w = np.zeros(len(x))
        w[0] = np.linalg.norm(x)
        # avoid substract nearly equal numbers
        if x[0] > 0:
            w[0] = -w[0]
        v = w - x
        H_num = np.matmul(v.reshape((len(v), 1)), v.reshape((1, len(v))))
        H_den = np.matmul(v.reshape((1, len(v))), v.reshape((len(v), 1)))
        H = np.identity(len(x)) - 2 * H_num / H_den
        I = np.identity(dim[0])
        I[j:, j:] = H
        # the householder reflector
        H = np.copy(I)
        # append this reflector to the list of reflectors
        householders.append(H)
        # update A
        A = np.matmul(H, A)
    # calculate Q, which is the product of all householder reflectors
    Q = householders[0]
    # loop through the list of householder reflectors
    for i in range(1, len(householders)):
        Q = np.matmul(Q, householders[i])
    return Q, A
