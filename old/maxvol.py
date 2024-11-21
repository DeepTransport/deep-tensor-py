import numpy as np
from numpy.linalg import det, inv, norm


np.random.seed(0)


def maxvol(A: np.ndarray, delta=1e-2):
    """Implementation of the max-volume algorithm in Goreinov et al. 
    (2010).
    """

    n, r = A.shape 

    eye = np.eye(n)

    # Confirm that the initial submatrix is full rank
    if np.abs(det(A[:r])) < 1e-8:
        raise Exception("Initial submatrix is singular.")

    # Compute B
    B = A @ inv(A[:r])

    I = np.arange(n)

    while True:

        # Find entry of B with greatest absolute value
        i_max, j_max = np.unravel_index(np.argmax(np.abs(B), axis=None), B.shape)
        b_ij = B[i_max, j_max]

        # Check for convergence
        if np.abs(b_ij) < 1 + delta:
            return I[:r]
        
        # Update indices of max vol matrix
        # A[[i_max, j_max], :] = A[[j_max, i_max], :]
        I[[i_max, j_max]] = I[[j_max, i_max]]

        # Make rank 1 update to B matrix
        B1 = (B[r:][:, j_max] + eye[r:][:, i_max])[:, np.newaxis]
        B2 = (B[i_max, :] - eye[j_max, :r])[np.newaxis, :]

        B[r:] -= B1 @ B2 / b_ij


def low_rank_approx(A: np.ndarray, r, delta=1e-2):
    """The low-rank approximation algorithm for matrices (Algorithm 2 in
    Oseledets and Tyrtyshnikov, 2010).
    """

    n, m = A.shape

    # Initialisation
    J = np.arange(r)

    k = 0
    A_k = np.zeros((n, m))
    As = [A_k]

    while True:

        # Carry out column cycle
        R = A[:, J]
        Q, T = np.linalg.qr(R)
        
        # Find max-volume submatrix in column R
        I = maxvol(Q)

        # Carry out row cycle
        C = A[I, :].T 
        Q, T = np.linalg.qr(C)

        # Get indices of Q
        J = maxvol(Q)

        Q_hat = Q[J, :]

        A_k = A[:, J] @ (Q @ inv(Q_hat)).T
        As.append(A_k)

        if norm(As[-1] - As[-2]) <= delta * norm(As[-1]):
            return As[-1]


A = np.random.rand(10, 10)
A_approx = low_rank_approx(A, 8)

print(A-A_approx)

print(np.linalg.matrix_rank(A))
print(np.linalg.matrix_rank(A_approx))
