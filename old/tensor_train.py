import numpy as np


class Tensor():

    def __init__(self, A: np.ndarray):

        self.A = A
        self.ns = A.shape
        self.nd = len(self.ns)
        self.nrm = np.linalg.norm(A)
    
    def _compute_tsvd(self, U, S, Vt, tol=0.0):
        for i in range(1, len(S)):
            if np.sum(np.square(S[i:])) <= tol**2:
                return U[:, :i], S[:i], Vt[:i, :]
        return U, S, Vt

    def compress_svd(self, A, eps):
        """Implements the TT-SVD algorithm (Algorithm 1 in 
        Oseledets and Tyrtyshnikov, 2010).
        """

        self.Gs = []
        self.rs = []

        # Compute desired tolerance
        tol = (eps * self.nrm) / np.sqrt(self.nd - 1)

        M = np.copy(A)

        r = 1
        self.rs.append(r)

        for k in range(self.nd-1):
            
            # Construct unfolding matrix
            M = np.reshape(M, (r*self.ns[k], -1))

            # Compute TSVD of unfolding matrix
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            U, S, Vt = self._compute_tsvd(U, S, Vt, tol)

            # Add rank of unfolding matrix to list of ranks
            r = len(S)
            self.rs.append(r)

            # Form tensor core
            G = np.reshape(U, (self.rs[k], self.ns[k], self.rs[k+1]))
            self.Gs.append(G)

            M = np.diag(S) @ Vt

        # Create final core
        G = M[:, :, np.newaxis]
        self.Gs.append(G)


    def _compute_tt_svd_old(self, A):
        """Implements the TT-SVD algorithm outlined by Oseledets 
        (2011).
        """

        # Initialise tensor to be decomposed
        C = np.copy(A)

        # Initialise list of cores
        self.Gs = []

        # Initialise list of core dimensions
        self.rs = np.zeros((self.nd, ), dtype=np.int64)
        self.rs[0] = 1

        for k in range(1, self.nd):
            
            # Reshape C to form appropriate unfolding matrix
            shape = (self.rs[k-1]*self.ns[k], -1)
            C = np.reshape(C, shape)

            # Compute SVD of reshaped tensor
            U, S, Vt = np.linalg.svd(C, full_matrices=False)

            self.rs[k] = len(S)

            shape = (self.rs[k-1], self.ns[k], self.rs[k])
            G = np.reshape(U, shape)
            self.Gs.append(G)

            C = np.diag(S) @ Vt

        self.Gs.append(C[:, :, np.newaxis])

    def __getitem__(self, inds):
        """Returns the element of a tensor at a specified index.
        """

        if len(inds) != self.nd:
            msg = ("Length of indices does not match "
                   + "number of dimensions of tensor.")
            raise Exception(msg)

        # Extract the relevant set of TT cores
        G_is = [self.Gs[i][:, inds[i], :] for i in range(self.nd)]

        # Take the product of the cores
        v = np.linalg.multi_dot(G_is)
        return v.flat[0]
