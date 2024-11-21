import numpy as np


np.random.seed(0)


def tt_svd(A: np.ndarray):
    """Implements the TT-SVD algorithm (Oseledets, 2011).
    """

    # Extract size of each dimension of tensor
    ns = A.shape
    d = len(ns)

    # Temporary tensor
    C = np.copy(A)

    # Tensor cores
    Gs = []

    rs = [1]

    for k in range(1, d):

        C = np.reshape(C, (rs[k-1]*ns[k], -1))

        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        
        rs.append(len(S))

        # Form new core
        G = np.reshape(U, (rs[k-1], ns[k], rs[k]))
        Gs.append(G)
        
        C = np.diag(S) @ Vt

    Gs.append(C[:, :, np.newaxis])

    return Gs

A = np.random.rand(3, 3, 3)
Gs = tt_svd(A)

inds = [0, 1, 2]

G_i = Gs[0][:, inds[0], :]

for i, ind in enumerate(inds[1:]):

    G_i = G_i @ Gs[i+1][:, ind, :]

print(G_i[0, 0])
print(A[0, 1, 2])