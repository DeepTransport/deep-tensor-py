from typing import Callable, Tuple

import torch

from .abstract_irt import AbstractIRT
from ..ftt import ApproxBases, Direction, InputData, TTData, TTFunc
from ..options import TTOptions
from ..polynomials import CDF1D, construct_cdf
from ..tools import reshape_matlab


def batch_mul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batch-multiplies two sets of matrices together.
    """
    return torch.einsum("...ij, ...jk", A, B)


class TTSIRT(AbstractIRT):

    def __init__(
        self, 
        potential: Callable[[torch.Tensor], torch.Tensor], 
        bases: ApproxBases,
        approx: TTFunc|None=None,
        options: TTOptions|None=None, 
        input_data: InputData|None=None, 
        tt_data: TTData|None=None,
        tau: float=1e-8
    ):
        
        if options is None:
            options = TTOptions()
        
        if input_data is None:
            input_data = InputData()
        
        def target_func(ls: torch.Tensor) -> torch.Tensor:
            """Returns the square root of the ratio between the target 
            density and the weighting function evaluated at a set of 
            points in the local domain ([-1, 1]^d).
            """
            return self.potential2density(potential, ls)
        
        AbstractIRT.__init__(
            self,
            potential, 
            bases,
            approx, 
            options, 
            input_data,
            tt_data
        )

        # Define coefficient tensors and marginalisation coefficents
        self.Bs: dict[int, torch.Tensor] = {}
        self.Rs: dict[int, torch.Tensor] = {} 
        
        self.int_dir = Direction.FORWARD
        self.tau = tau

        self.approx = self.build_approximation(
            target_func, 
            bases,
            options, 
            input_data,
            tt_data
        )

        self._oned_cdfs = {}
        for k in range(self.dim):
            self._oned_cdfs[k] = construct_cdf(
                poly=self.approx.bases.polys[k], 
                error_tol=self.approx.options.cdf_tol
            )

        self.marginalise(direction=self.int_dir)
        return

    @property 
    def oned_cdfs(self) -> dict[int, CDF1D]:
        return self._oned_cdfs

    @property
    def approx(self) -> TTFunc:
        return self._approx

    @approx.setter 
    def approx(self, value: TTFunc):
        self._approx = value
        return

    @property
    def int_dir(self) -> Direction:
        return self._int_dir
    
    @int_dir.setter
    def int_dir(self, value: Direction):
        self._int_dir = value
        return

    @property 
    def tau(self) -> torch.Tensor:
        return self._tau
    
    @tau.setter
    def tau(self, value: torch.Tensor):
        self._tau = value 
        return
    
    @property 
    def z(self) -> torch.Tensor:
        return self._z 
    
    @z.setter
    def z(self, value: torch.Tensor):
        self._z = value
        return

    @property 
    def z_func(self) -> torch.Tensor:
        return self._z_func

    @z_func.setter
    def z_func(self, value: torch.Tensor):
        self._z_func = value
        return

    def _marginalise_forward(self) -> None:
        """Computes each coefficient tensor required to evaluate the 
        marginal functions in each dimension, by iterating over the 
        dimensions of the approximation from last to first.
        """

        self.Rs[self.dim] = torch.tensor([[1.0]])

        for k in range(self.dim-1, -1, -1):
            
            poly = self.approx.bases.polys[k]
            A = self.approx.data.cores[k]

            self.Bs[k] = torch.einsum("ijl, lk", A, self.Rs[k+1])
            C_k = torch.einsum("ilk, lj", self.Bs[k], poly.mass_R)
            C_k = self.approx.unfold_right(C_k)
            self.Rs[k] = torch.linalg.qr(C_k, mode="reduced")[1].T

        self.z_func = self.Rs[0].square().sum()
        self.z = self.z_func + self.tau
        return 
    
    def _marginalise_backward(self) -> None:
        """Computes each coefficient tensor required to evaluate the 
        marginal functions in each dimension, by iterating over the 
        dimensions of the approximation from first to last.
        """
        
        self.Rs[-1] = torch.tensor([[1.0]])

        for k in range(self.dim):
            
            poly = self.approx.bases.polys[k]
            A = self.approx.data.cores[k]

            self.Bs[k] = torch.einsum("il, ljk", self.Rs[k-1], A)
            C_k = torch.einsum("jl, ilk", poly.mass_R, self.Bs[k])
            C_k = self.approx.unfold_left(C_k)
            self.Rs[k] = torch.linalg.qr(C_k, mode="reduced")[1]

        self.z_func = self.Rs[self.dim-1].square().sum()
        self.z = self.z_func + self.tau
        return

    def marginalise(
        self, 
        direction: Direction=Direction.FORWARD
    ) -> None:
        """Computes each coefficient tensor (B_k) required to evaluate 
        the marginal functions in each dimension, as well as the 
        normalising constant, z. 

        Parameters
        ----------
        direction:
            The direction in which to iterate over the tensor cores.

        Returns
        -------
        None

        Notes
        -----
        Updates self.Bs, self.z_func, self.z.

        References
        ----------
        Cui and Dolgov (2022, Sec. 3.1). Deep composition of tensor 
        trains using squared inverse Rosenblatt transports.

        """
        self.int_dir = direction
        if self.int_dir == Direction.FORWARD:
            self._marginalise_forward()
        else:
            self._marginalise_backward()
        return

    def _eval_irt_local_forward(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the inverse Rosenblatt transport by iterating over
        the dimensions from first to last.

        Parameters
        ----------
        zs:
            An n * d matrix of samples from [0, 1]^d.

        Returns
        -------
        ls: 
            An n * d matrix containing a set of samples from the local 
            domain, obtained by applying the IRT to each sample in zs.
        ps_sq:
            An n-dimensional vector containing the square of the FTT 
            approximation to the square root of the target function, 
            evaluated at each sample in zs.
        
        """

        n_zs, d_zs = zs.shape
        ls = torch.zeros_like(zs)
        ps = torch.ones((n_zs, 1))

        for k in range(d_zs):

            r_p, _, r_k = self.approx.data.cores[k].shape
            n_k = self.oned_cdfs[k].cardinality
            
            Gs_cdf = TTFunc.eval_oned_core_213(
                self.bases.polys[k], 
                self.Bs[k], 
                self.oned_cdfs[k].nodes
            ).reshape(n_k, r_p, r_k)

            gls = torch.einsum("jl, ilk -> ijk", ps, Gs_cdf)
            gls_sq = gls.square().sum(dim=2)

            ls[:, k] = self.oned_cdfs[k].invert_cdf(
                gls_sq + self.tau, 
                zs[:, k]
            )

            Gs = TTFunc.eval_oned_core_213(
                self.bases.polys[k], 
                self.approx.data.cores[k], 
                ls[:, k]
            ).reshape(n_zs, r_p, r_k)

            ps = torch.einsum("il, ilk -> ik", ps, Gs)
        
        ps_sq = (ps @ self.Rs[d_zs]).square().sum(dim=1)
        return ls, ps_sq
    
    def _eval_irt_local_backward(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the inverse Rosenblatt transport by iterating over
        the dimensions from last to first.

        Parameters
        ----------
        zs:
            An n * d matrix of samples from [0, 1]^d.

        Returns
        -------
        ls: 
            An n * d matrix containing a set of samples from the local 
            domain, obtained by applying the IRT to each sample in zs.
        ps_sq:
            An n-dimensional vector containing the square of the FTT 
            approximation to the square root of the target function, 
            evaluated at each sample in zs.
        
        """

        n_zs, d_zs = zs.shape
        ls = torch.zeros_like(zs)
        ps = torch.ones((n_zs, 1))
        i_min = self.dim - d_zs
        
        for k in range(self.dim-1, i_min-1, -1):
            
            k_ind = k - i_min
            r_p, _, r_k = self.approx.data.cores[k].shape
            n_k = self.oned_cdfs[k].cardinality

            Gs_cdf = TTFunc.eval_oned_core_231(
                self.bases.polys[k],
                self.Bs[k],
                self.oned_cdfs[k].nodes
            ).reshape(n_k, r_k, r_p)

            gls = torch.einsum("ilk, jl -> ijk", Gs_cdf, ps)
            gls_sq = gls.square().sum(dim=2)

            ls[:, k_ind] = self.oned_cdfs[k].invert_cdf(
                gls_sq + self.tau, 
                zs[:, k_ind]
            )

            Gs = TTFunc.eval_oned_core_231(
                self.bases.polys[k],
                self.approx.data.cores[k],
                ls[:, k_ind]
            ).reshape(n_zs, r_k, r_p)

            ps = torch.einsum("il, ilk -> ik", ps, Gs)

        ps_sq = (self.Rs[i_min-1] @ ps.T).square().sum(dim=0)
        return ls, ps_sq

    def eval_irt_local(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts a set of realisations of a standard uniform 
        random variable, Z, to the corresponding realisations of the 
        local (i.e., defined on [-1, 1]) target random variable, by 
        applying the inverse Rosenblatt transport.
        
        Parameters
        ----------
        zs: 
            An n * d matrix containing values on [0, 1]^d.

        Returns
        -------
        ls:
            An n * d matrix containing the corresponding samples of the 
            target random variable mapped into the local domain.
        neglogfls:
            The local potential function associated with the 
            approximation to the target density, evaluated at each 
            sample.

        """

        if self.int_dir == Direction.FORWARD:
            ls, ps_sq = self._eval_irt_local_forward(zs)
        else:
            ls, ps_sq = self._eval_irt_local_backward(zs)
        
        indices = self.get_transform_indices(zs.shape[1])
        
        neglogpls = -(ps_sq + self.tau).log()
        neglogwls = self.bases.eval_measure_potential_local(ls, indices)
        neglogfls = self.z.log() + neglogpls + neglogwls

        return ls, neglogfls
    
    def get_potential2density(
        self, 
        ys: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        
        raise NotImplementedError()

    def potential2density(
        self, 
        potential_func: Callable[[torch.Tensor], torch.Tensor], 
        ls: torch.Tensor
    ) -> torch.Tensor:
        """Returns the square root of the target function evaluated at
        a set of samples in the local domain.
        
        Parameters
        ----------
        ls:
            An n * d matrix containing a set of n samples from the 
            local domain ([-1, 1]^d).
        potential_func:
            A function that evaluates the potential (negative log) 
            of the target function at a given set of samples from the 
            approximation domain.

        Returns
        -------
        ps:
            An n-dimensional vector containing the square root of the 
            ratio of the potential function and the weighting function, 
            evaluated at each element of ls.

        """
        
        xs = self.bases.local2approx(ls)[0]
        neglogfxs = potential_func(xs)
        neglogwxs = self.bases.eval_measure_potential(xs)[0]

        # The ratio of f and w is invariant to changes of coordinate
        ps = torch.exp(-0.5 * (neglogfxs - neglogwxs))
        return ps

    def build_approximation(
        self, 
        target_func: Callable[[torch.Tensor], torch.Tensor], 
        bases: ApproxBases, 
        options: TTOptions, 
        input_data: InputData,
        tt_data: TTData
    ) -> TTFunc:
        
        approx = TTFunc(
            target_func, 
            bases,
            options=options, 
            input_data=input_data,
            tt_data=tt_data
        )
        approx.cross()

        if approx.use_amen:
            approx.round()

        return approx

    def eval_potential_local(
        self, 
        ls: torch.Tensor
    ) -> torch.Tensor:

        dim_l = ls.shape[1]

        if self.int_dir == Direction.FORWARD:

            gs = self.approx.eval_local(ls, direction=self.int_dir)

            if dim_l < self.dim:
                ps_sq = (gs @ self.Rs[dim_l]).square().sum(dim=1)
            else: 
                ps_sq = gs.square()

            indices = torch.arange(dim_l)
            
        else:

            gs = self.approx.eval_local(ls, direction=self.int_dir)
            i_min = self.dim - dim_l

            if dim_l < self.dim:
                ps_sq = (self.Rs[i_min-1] @ gs).square().sum(dim=0)
            else:
                ps_sq = gs.square()

            indices = torch.arange(self.dim-1, self.dim-dim_l-1, -1)
            
        # TODO: check that indices go backwards. This could be an issue 
        # if different bases are used in each dimension.
        neglogwls = self.approx.bases.eval_measure_potential_local(ls, indices)
        neglogfls = self.z.log() - (ps_sq + self.tau).log() + neglogwls
        return neglogfls

    def _eval_rt_local_forward(
        self, 
        ls: torch.Tensor
    ) -> torch.Tensor:

        n_ls, d_ls = ls.shape
        zs = torch.zeros_like(ls)
        Gs_prod = torch.ones((n_ls, 1))
            
        for k in range(d_ls):
            
            r_p, _, r_k = self.approx.data.cores[k].shape
            n_k = self.oned_cdfs[k].cardinality
            
            # Compute (unnormalised) conditional PDF for each sample
            Ps = TTFunc.eval_oned_core_213(
                self.bases.polys[k],
                self.Bs[k],
                self.oned_cdfs[k].nodes
            ).reshape(n_k, r_p, r_k)
            gs = torch.einsum("jl, ilk -> ijk", Gs_prod, Ps)
            ps = gs.square().sum(dim=2) + self.tau

            # Evaluate CDF to obtain corresponding uniform variates
            zs[:, k] = self.oned_cdfs[k].eval_cdf(ps, ls[:, k])

            # Compute incremental product of tensor cores for each sample
            Gs = TTFunc.eval_oned_core_213(
                self.bases.polys[k], 
                self.approx.data.cores[k], 
                ls[:, k]
            ).reshape(n_ls, r_p, r_k)
            Gs_prod = torch.einsum("il, ilk -> ik", Gs_prod, Gs)

        return zs
    
    def _eval_rt_local_backward(
        self, 
        ls: torch.Tensor
    ) -> torch.Tensor:

        n_ls, d_ls = ls.shape
        zs = torch.zeros_like(ls)
        k_min = self.dim - d_ls
        Gs_prod = torch.ones((1, n_ls))

        for k in range(self.dim-1, k_min-1, -1):
            
            k_ind = k - k_min
            r_p, _, r_k = self.approx.data.cores[k].shape
            n_k = self.oned_cdfs[k].cardinality

            # Compute (unnormalised) conditional PDF for each sample
            Ps = self.approx.eval_oned_core_213(
                self.bases.polys[k],
                self.Bs[k],
                self.oned_cdfs[k].nodes
            ).reshape(n_k, r_p, r_k)
            ps_sq = torch.einsum("ijl, lk -> ijk", Ps, Gs_prod)
            ps_sq = ps_sq.square().sum(dim=1)

            # Evaluate CDF to obtain corresponding uniform variates
            zs[:, k_ind] = self.oned_cdfs[k].eval_cdf(
                ps_sq + self.tau, 
                ls[:, k_ind]
            )
            
            # Compute incremental product of tensor cores for each sample
            Gs = TTFunc.eval_oned_core_213(
                self.bases.polys[k], 
                self.approx.data.cores[k], 
                ls[:, k_ind]
            ).reshape(n_ls, r_p, r_k)
            Gs_prod = torch.einsum("ijl, li -> ji", Gs, Gs_prod)

        return zs

    def eval_rt_local(
        self, 
        ls: torch.Tensor
    ) -> torch.Tensor:
        if self.int_dir == Direction.FORWARD:
            zs = self._eval_rt_local_forward(ls)
        else:
            zs = self._eval_rt_local_backward(ls)
        return zs

    def eval_irt_local_grad(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates the inverse of the squared Rosenblatt transport
        X = R^{-1}(Z), where X is the target random variable and Z is
        uniform.
        
        Parameters
        ----------
        zs: 
            An n * d matrix containing variates from the standard 
            uniform distribution.

        Returns
        -------
        %   X - random variable drawn from the pdf defined by SIRT
        %   f - potential function at X
        %   g - gradient of potential function at X

        """

        # TODO: figure out what to do with this function -- there are 
        # two distinct versions, one which doesn't return the gradient 
        # and can be used for the marginals, and one which does return 
        # the gradient but cannot be used for the marginals
        num_z, dim_z = zs.shape

        if dim_z != self.dim:
            msg = "Grad not implemented for marginals."
            raise Exception(msg)
        
        rs = torch.zeros_like(zs)

        fls = {}
        frs = {}

        if self.int_dir > 0:  # TODO: make a new direction thingy
            
            frl = torch.ones(num_z)

            for k in range(num_z):

                # TODO: give these more descriptive names
                rkm = self.approx.data.cores[k].shape[0]
                num_nodes = self.oned_cdfs[k].cardinality

                T1 = self.approx.eval_oned_core_213(
                    self.approx.bases.polys[k], 
                    self.Bs[k], 
                    self.oned_cdfs[k].nodes
                )
                T1 = reshape_matlab(T1, (rkm, -1))

                pk = reshape_matlab(frl @ T1, (num_z * num_nodes, -1)) ** 2
                pk = reshape_matlab(torch.sum(pk, dim=1), (num_z, num_nodes))

                rs[:, k] = self.oned_cdfs[k].invert_cdf(pk + self.tau, zs[:, k])

                T2 = self.approx.eval_oned_core_213(
                    self.approx.bases.polys[k], 
                    self.approx.data.cores[k], 
                    rs[:, k]
                )

                # TODO
                ii = 1.0
                jj = torch.arange()
                raise NotImplementedError("TODO")

        else:  # from right to left
            raise NotImplementedError("TODO")
    
    def _eval_cirt_local_forward(
        self, 
        ls_x: torch.Tensor, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # from the last dimension, marginalise to the first
        # order of the samples (x, r)
        # first evaluate the marginal for x

        num_x, dim_x = ls_x.shape
        num_z, dim_z = zs.shape

        ls_y = torch.zeros_like(zs)
        
        frl = torch.ones(num_z, 1)

        for k in range(dim_x-1):
            
            rank_p = self.approx.data.cores[k].shape[0]
            
            T2 = self.approx.eval_oned_core_213(
                self.approx.bases.polys[k],
                self.approx.data.cores[k],
                ls_x[:, k]
            )

            ii = torch.arange(num_x).repeat(rank_p)
            jj = (torch.arange(rank_p * num_x)
                    .reshape(num_x, rank_p).T
                    .flatten())
            indices = torch.vstack((ii[None, :], jj[None, :]))
            values = frl.T.flatten()
            size = (num_x, rank_p * num_x)

            B = torch.sparse_coo_tensor(indices, values, size)

            frl = B @ T2

        rank_p = self.approx.data.cores[dim_x-1].shape[0]

        T2 = self.approx.eval_oned_core_213(
            self.approx.bases.polys[dim_x-1], 
            self.Bs[dim_x-1], 
            ls_x[:, dim_x-1]
        )

        ii = torch.arange(num_x).repeat(rank_p)
        jj = (torch.arange(rank_p * num_x)
                .reshape(num_x, rank_p).T
                .flatten())
        indices = torch.vstack((ii[None, :], jj[None, :]))
        values = frl.T.flatten()
        size = (num_x, rank_p * num_x)
        B = torch.sparse_coo_tensor(indices, values, size)
        
        frl_m = B @ T2
        fm = frl_m.square().sum(dim=1)
        
        T2 = self.approx.eval_oned_core_213(
            self.approx.bases.polys[dim_x-1], 
            self.approx.data.cores[dim_x-1], 
            ls_x[:, dim_x-1]
        )

        frl = B @ T2

        # Generate conditional samples
        for j in range(dim_z):
            
            k = dim_x + j

            rank_p = self.approx.data.cores[k].shape[0]
            num_nodes = self.oned_cdfs[k].nodes.numel()

            T1 = self.approx.eval_oned_core_213(
                self.approx.bases.polys[k], 
                self.Bs[k], 
                self.oned_cdfs[k].nodes
            ).T.reshape(-1, rank_p).T

            pk = ((frl @ T1).T
                  .reshape(-1, num_z*num_nodes)
                  .square()
                  .sum(dim=0)
                  .reshape(num_nodes, num_z))

            ls_y[:, j] = self.oned_cdfs[k].invert_cdf(pk+self.tau, zs[:, j])

            T2 = self.approx.eval_oned_core_213(
                self.approx.bases.polys[k], 
                self.approx.data.cores[k],
                ls_y[:, j]
            )

            ii = torch.arange(num_z).repeat(rank_p)
            jj = (torch.arange(rank_p * num_z)
                    .reshape(num_x, rank_p).T
                    .flatten())
            indices = torch.vstack((ii[None, :], jj[None, :]))
            values = frl.T.flatten()
            size = (num_z, rank_p * num_z)
            B = torch.sparse_coo_tensor(indices, values, size)
            frl = B @ T2

        fs = frl.flatten().square()

        indices = dim_x + torch.arange(dim_z)
        neglogwls_y = self.approx.bases.eval_measure_potential_local(ls_y, indices)
        neglogfls_y = (fm + self.tau).log() - (fs + self.tau).log() + neglogwls_y

        return ls_y, neglogfls_y
    
    def _eval_cirt_local_backward(
        self, 
        ls_x: torch.Tensor, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        num_x, dim_x = ls_x.shape
        num_z, dim_z = zs.shape

        ls_y = torch.zeros_like(zs)
        
        frg = torch.ones(num_z, 1)

        for j in range(dim_x-1, 0, -1):
            
            k = dim_z + j
            rank_k = self.approx.data.cores[k].shape[-1]
            
            T2 = self.approx.eval_oned_core_231(
                self.approx.bases.polys[k],
                self.approx.data.cores[k],
                ls_x[:, j]
            )

            ii = torch.arange(rank_k * num_x)
            jj = torch.arange(num_x).repeat_interleave(rank_k)
            indices = torch.vstack((ii[None, :], jj[None, :]))
            values = frg.T.flatten()
            size = (rank_k * num_x, num_x)

            B = torch.sparse_coo_tensor(indices, values, size)
            frg = T2.T @ B

        rank_k = self.approx.data.cores[dim_z].shape[-1]

        T2 = self.approx.eval_oned_core_231(
            self.approx.bases.polys[dim_z], 
            self.Bs[dim_z], 
            ls_x[:, 0]
        )

        ii = torch.arange(rank_k * num_x)
        jj = torch.arange(num_x).repeat_interleave(rank_k)
        indices = torch.vstack((ii[None, :], jj[None, :]))
        values = frg.T.flatten()
        size = (rank_k * num_x, num_x)
        B = torch.sparse_coo_tensor(indices, values, size)
        
        frg_m = T2.T @ B
        fm = frg_m.square().sum(dim=0)
        
        T2 = self.approx.eval_oned_core_231(
            self.approx.bases.polys[dim_z], 
            self.approx.data.cores[dim_z], 
            ls_x[:, 0]
        )

        frg = T2.T @ B

        # Generate conditional samples
        for k in range(dim_z-1, -1, -1):

            rank_k = self.approx.data.cores[k].shape[-1]
            num_nodes = self.oned_cdfs[k].nodes.numel()

            T1 = reshape_matlab(
                self.approx.eval_oned_core_213(
                    self.approx.bases.polys[k], 
                    self.Bs[k], 
                    self.oned_cdfs[k].nodes
                ), 
                (-1, rank_k)
            )

            pk = reshape_matlab(
                reshape_matlab(T1 @ frg, (-1, num_z*num_nodes)).square().sum(dim=0), 
                (num_nodes, num_z)
            )

            ls_y[:, k] = self.oned_cdfs[k].invert_cdf(pk+self.tau, zs[:, k])

            T2 = self.approx.eval_oned_core_231(
                self.approx.bases.polys[k], 
                self.approx.data.cores[k],
                ls_y[:, k]
            )

            ii = torch.arange(rank_k * num_z)
            jj = torch.arange(num_z).repeat_interleave(rank_k)
            indices = torch.vstack((ii[None, :], jj[None, :]))
            values = frg.T.flatten()
            size = (rank_k * num_z, num_z)
            B = torch.sparse_coo_tensor(indices, values, size)
            frg = T2.T @ B

        fs = frg.flatten().square()

        indices = torch.arange(dim_z-1, -1, -1)
        neglogwls_y = self.approx.bases.eval_measure_potential_local(ls_y, indices)
        neglogfls_y = (fm + self.tau).log() - (fs + self.tau).log() + neglogwls_y

        return ls_y, neglogfls_y

    def eval_cirt_local(
        self, 
        ls_x: torch.Tensor, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.int_dir == Direction.FORWARD:
            ls_y, neglogfls_y = self._eval_cirt_local_forward(ls_x, zs)
        else:
            ls_y, neglogfls_y = self._eval_cirt_local_backward(ls_x, zs)

        return ls_y, neglogfls_y
    
    def _eval_rt_jac_local_forward_alt(
        self, 
        ls: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:

        block_ftt: dict[int, torch.Tensor] = {}
        block_marginal: dict[int, torch.Tensor] = {}
        neglogwls: dict[int, torch.Tensor] = {}
        Ts: dict[int, torch.Tensor] = {}
        block_ftt_deriv: dict[int, torch.Tensor] = {}

        n_ls = ls.shape[0]
        Js = torch.zeros((self.dim, n_ls, self.dim))

        for k in range(self.dim):
            
            r_p, _, r_k = self.approx.data.cores[k].shape

            block_ftt[k] = TTFunc.eval_oned_core_213(
                self.bases.polys[k],
                self.approx.data.cores[k],
                ls[:, k]
            ).reshape(n_ls, r_p, r_k)

            block_ftt_deriv[k] = TTFunc.eval_oned_core_213_deriv(
                self.bases.polys[k], 
                self.approx.data.cores[k],
                ls[:, k]
            ).reshape(n_ls, r_p, r_k)

            block_marginal[k] = TTFunc.eval_oned_core_213(
                self.bases.polys[k], 
                self.Bs[k],
                ls[:, k]
            ).reshape(n_ls, r_p, r_k)

            Ts[k] = TTFunc.eval_oned_core_213(
                self.bases.polys[k],
                self.Bs[k],
                self.oned_cdfs[k].nodes
            ).T.reshape(-1, r_p).T 

            neglogwls[k] = -self.bases.polys[k].eval_log_measure(ls[:, k])

        Fs: dict[int, torch.Tensor] = {}  # accumulated FTT 
        Gs: dict[int, torch.Tensor] = {}  # sum(G**2) is the marginal, each G{k} is nxr
        Fm: dict[int, torch.Tensor] = {}  # accumulated FTT part II

        Fm[-1] = self.z

        Fs[0] = block_ftt[0].clone()
        Gs[0] = block_marginal[0].clone()
        Fm[0] = Gs[0].square().sum(dim=(1, 2)) + self.tau

        for k in range(1, self.dim):
            Fs[k] = torch.einsum("...ij, ...jk", Fs[k-1], block_ftt[k])
            Gs[k] = torch.einsum("...ij, ...jk", Fs[k-1], block_marginal[k])
            Fm[k] = Gs[k].square().sum(dim=(1, 2)) + self.tau

        for j in range(self.dim):

            r_p, _, r_k = self.approx.data.cores[j].shape
            
            # Fill in diagonal elements
            Js[j, :, j] = torch.exp(-neglogwls[j]) * Fm[j] / Fm[j-1]

            if j < self.dim-1:  # skip the (d, d) element
                
                # Derivative of the FTT
                drl = block_ftt_deriv[j].clone()

                # Derivative of the FTT, 2nd term, for the d(j+1)/dj term
                mrl = TTFunc.eval_oned_core_213_deriv(
                    self.approx.bases.polys[j], 
                    self.Bs[j], 
                    ls[:, j]
                ).reshape(n_ls, r_p, r_k)

                if j > 0:
                    drl = torch.einsum("...ij, ...jk", Fs[j-1], drl)
                    mrl = torch.einsum("...ij, ...jk", Fs[j-1], mrl)

                # First sub, the second term, for the d(j+1)/dj term
                Js[j+1, :, j] -= 2 * torch.sum(Gs[j] * mrl, dim=(1, 2)) * zs[:, j+1]

                for k in range(j+1, self.dim):
                    
                    # Accumulate the j-th block and evaluate the integral
                    r_p, _, r_k = self.approx.data.cores[k].shape
                    n_k = self.oned_cdfs[k].cardinality

                    pk = (Fs[k-1].reshape(n_ls, r_p) @ Ts[k]) * (drl.reshape(n_ls, r_p) @ Ts[k])
                    pk = pk.T.reshape(-1, n_ls * n_k).sum(dim=0).reshape(n_k, n_ls)

                    # Compute the first term
                    if self.bases.polys[k].constant_weight:
                        wls = self.bases.polys[k].eval_measure(self.oned_cdfs[k].nodes)
                        pk *= wls[:, None]

                    # TODO: do a finite difference check for this?
                    Js[k, :, j] += 2 * self.oned_cdfs[k].eval_int_deriv(pk, ls[:, k])
                    Js[k, :, j] /= Fm[k-1]

                    if k < self.dim-1:
                        # the second term, for the d(k+1)/dj term
                        mrl = torch.einsum("...ij, ...jk", drl, block_marginal[k])
                        # accumulate
                        drl = torch.einsum("...ij, ...jk", drl, block_ftt[k])

                        Js[k+1, :, j] -= 2 * (Gs[k] * mrl).sum(dim=(1, 2)) * zs[:, k+1]
        
        Js = Js.reshape(self.dim, self.dim * n_ls) # TEMP
        return Js
    
    def eval_rt_jac_local(
        self, 
        ls: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the Jacobian of the Rosenblatt transport.
        
        Parameters
        ----------
        ls:
            An n * d set of samples from the local domain.
        zs: 
            An n * d matrix corresponding to evaluations of the 
            Rosenblatt transport at each sample in ls.
        
        Returns
        -------
        Js:
            A d * (d*n) matrix, where each d * d block contains the 
            Jacobian of the Rosenblatt transport evaluated at a given 
            sample: that is, J_ij = dz_i / dl_i.

        """

        TTFunc._check_sample_dim(ls, self.dim, strict=True)

        if self.int_dir == Direction.FORWARD:
            J = self._eval_rt_jac_local_forward(ls)

        else:

            # TODO: eventually layer this
            n_ls = ls.shape[0]
            J = torch.zeros((self.dim, self.dim * n_ls))

            block_ftt: dict[int, torch.Tensor] = {}
            block_marginal: dict[int, torch.Tensor] = {}
            neglogwls: dict[int, torch.Tensor] = {}
            Ts: dict[int, torch.Tensor] = {}
            block_ftt_deriv: dict[int, torch.Tensor] = {}

            for k in range(self.dim):
                
                block_ftt[k] = TTFunc.eval_oned_core_231(
                    self.bases.polys[k], 
                    self.approx.data.cores[k],
                    ls[:, k]
                )

                block_marginal[k] = TTFunc.eval_oned_core_231(
                    self.bases.polys[k], 
                    self.Bs[k],
                    ls[:, k]
                )

                r_k = self.approx.data.cores[k].shape[-1]
                Ts[k] = reshape_matlab(TTFunc.eval_oned_core_213(self.bases.polys[k], self.Bs[k], self.oned_cdfs[k].nodes), (-1, r_k))
                block_ftt_deriv[k] = TTFunc.eval_oned_core_231_deriv(self.bases.polys[k], self.approx.data.cores[k], ls[:, k])
                neglogwls[k] = -self.bases.polys[k].eval_log_measure(ls[:, k])

            Fs = {}
            Gs = {}
            Fs[self.dim-1] = block_ftt[self.dim-1].T.clone()
            Gs[self.dim-1] = block_marginal[self.dim-1].T.clone()

            for k in range(self.dim-2, -1, -1):
                r_k = self.approx.data.cores[k].shape[-1]
                
                ii = torch.arange(r_k*n_ls)
                jj = torch.arange(n_ls).repeat_interleave(r_k)
                indices = torch.vstack((ii[None, :], jj[None, :]))
                B = torch.sparse_coo_tensor(
                    indices=indices, 
                    values=Fs[k+1].T.flatten(), 
                    size=(r_k*n_ls, n_ls)
                )
                Fs[k] = block_ftt[k].T @ B 
                Gs[k] = block_marginal[k].T @ B 

            Fm = {}
            for k in range(self.dim):
                Fm[k] = Gs[k].square().sum(dim=0) + self.tau 
            
            for j in range(self.dim-1, -1, -1):
                inds = torch.arange(0, self.dim*n_ls, self.dim) + j 

                if j == self.dim-1:
                    J[j, inds] = Fm[j] / self.z
                else:
                    J[j, inds] = Fm[j] / Fm[j+1]
                
                J[j, inds] *= torch.exp(-neglogwls[j])

                if j > 0:  # skip the (1, 1) element 
                    
                    drg = block_ftt_deriv[j].T
                    mrg = TTFunc.eval_oned_core_231_deriv(
                        self.bases.polys[j], 
                        self.Bs[j], 
                        ls[:, j]
                    ).T

                    if j < self.dim-1:
                        r_j = self.approx.data.cores[j].shape[-1]
                        ii = torch.arange(r_j*n_ls)
                        jj = torch.arange(n_ls).repeat_interleave(r_j)
                        indices = torch.vstack((ii[None, :], jj[None, :]))
                        B = torch.sparse_coo_tensor(
                            indices=indices, 
                            values=Fs[j+1].T.flatten(), 
                            size=(r_j*n_ls, n_ls)
                        )
                        drg = drg @ B 
                        mrg = mrg @ B 
                    
                    J[j-1, inds] = J[j-1, inds] - 2 * torch.sum(Gs[j] * mrg, dim=0) * zs[:, j-1]

                    for k in range(j-1, -1, -1):
                        
                        r_k = self.approx.data.cores[k].shape[-1]
                        n_k = self.oned_cdfs[k].nodes.numel()
                        pk = reshape_matlab(torch.sum(reshape_matlab((Ts[k] @ Fs[k+1]) * (Ts[k] @ drg), (-1, n_k*n_ls)), dim=0), (n_k, n_ls))
                        
                        if self.bases.polys[k].constant_weight:
                            tmp = self.bases.polys[k].eval_measure(self.oned_cdfs[k].nodes)[:, None]
                            pk = pk * tmp
                        
                        J[k, inds] = J[k, inds] + 2 * reshape_matlab(self.oned_cdfs[k].eval_int_deriv(pk, ls[:, k]), (1, -1))

                        if k > 0:
                            ii = torch.arange(r_k*n_ls)
                            jj = torch.arange(n_ls).repeat_interleave(r_k)
                            indices = torch.vstack((ii[None, :], jj[None, :]))
                            B = torch.sparse_coo_tensor(
                                indices=indices, 
                                values=drg.T.flatten(), 
                                size=(r_k*n_ls, n_ls)
                            )
                            drg = block_ftt[k].T @ B 
                            mrg = block_marginal[k].T @ B 
                            J[k-1, inds] = J[k-1, inds] - 2 * torch.sum(Gs[k] * mrg, dim=0) * zs[:, k-1]
                        
                        J[k, inds] = J[k, inds] / Fm[k+1]

        return J

    def _eval_rt_jac_local_forward(
        self,
        ls: torch.Tensor
    ) -> torch.Tensor:
        
        n_ls = ls.shape[0]
        TTFunc._check_sample_dim(ls, self.dim, strict=True)

        Jacs = torch.zeros((self.dim, n_ls, self.dim))

        Gs = {}
        Gs_deriv = {}
        Ps = {}
        Ps_deriv = {}
        Ps_grid = {}

        ps_marg = {}
        ps_marg[-1] = self.z
        ps_marg_deriv = {}
        ps_grid = {}
        ps_grid_deriv = {}
        wls = {}

        gs = torch.ones((n_ls, 1, 1))
        Gs_prod = {} 
        Gs_prod[-1] = torch.ones((n_ls, 1, 1))

        for k in range(self.dim):

            r_p, _, r_k = self.approx.data.cores[k].shape
            n_k = self.oned_cdfs[k].cardinality

            # Evaluate kth tensor core
            Gs[k] = TTFunc.eval_oned_core_213(
                self.bases.polys[k], 
                self.approx.data.cores[k], 
                ls[:, k]
            ).reshape(n_ls, r_p, r_k)

            # Evaluate derivative of kth tensor core
            Gs_deriv[k] = TTFunc.eval_oned_core_213_deriv(
                self.bases.polys[k], 
                self.approx.data.cores[k], 
                ls[:, k]
            ).reshape(n_ls, r_p, r_k)

            # Evaluate kth marginalisation core at samples
            Ps[k] = TTFunc.eval_oned_core_213(
                self.bases.polys[k],
                self.Bs[k],
                ls[:, k]
            ).reshape(n_ls, r_p, r_k)

            # Evaluate derivative of kth marginalisation core at samples
            Ps_deriv[k] = TTFunc.eval_oned_core_213_deriv(
                self.bases.polys[k],
                self.Bs[k],
                ls[:, k]
            ).reshape(n_ls, r_p, r_k)

            # Evaluate kth marginalisation core at nodes of current CDF
            Ps_grid[k] = TTFunc.eval_oned_core_213(
                self.bases.polys[k],
                self.Bs[k],
                self.oned_cdfs[k].nodes
            ).reshape(n_k, r_p, r_k)

            Gs_prod[k] = batch_mul(Gs_prod[k-1], Gs[k])

        # Weighting function and marginal probabilities
        for k in range(self.dim):

            # Evaluate weighting function for current dimension
            wls[k] = self.bases.polys[k].eval_measure(ls[:, k])

            # Evaluate marginal probability for the first k elements of 
            # each sample
            gs = batch_mul(Gs_prod[k-1], Ps[k])
            ps_marg[k] = gs.square().sum(dim=(1, 2)) + self.tau

        # Off-diagonal stuff (marginal CDF on grid)
        for k in range(self.dim):
            # Compute (unnormalised) marginal PDF at each point on grid for each sample
            gs = torch.einsum("mij, ljk -> lmik", Gs_prod[k-1], Ps_grid[k])
            ps_grid[k] = gs.square().sum(dim=(2, 3)) + self.tau

        # Derivatives of marginal PDF
        for k in range(self.dim-1):
            ps_marg_deriv[k] = {}
            
            for j in range(k+1):

                prod = batch_mul(Gs_prod[k-1], Ps[k])
                prod_deriv = torch.ones((n_ls, 1, 1))

                for k_i in range(k):
                    core = Gs_deriv[k_i] if k_i == j else Gs[k_i]
                    prod_deriv = batch_mul(prod_deriv, core)
                core = Ps_deriv[k] if k == j else Ps[k]
                prod_deriv = batch_mul(prod_deriv, core)

                ps_marg_deriv[k][j] = 2 * (prod * prod_deriv).sum(dim=(1, 2))

        for k in range(1, self.dim):
            ps_grid_deriv[k] = {}

            for j in range(k):

                prod = torch.einsum("mij, ljk -> lmik", Gs_prod[k-1], Ps_grid[k])
                prod_deriv = torch.ones((n_ls, 1, 1))

                for k_i in range(k):
                    core = Gs_deriv[k_i] if k_i == j else Gs[k_i]
                    prod_deriv = batch_mul(prod_deriv, core)
                prod_deriv = torch.einsum("mij, ljk -> lmik", prod_deriv, Ps_grid[k])
                
                ps_grid_deriv[k][j] = 2 * (prod * prod_deriv).sum(dim=(2, 3))

        # Populate diagonal elements
        for k in range(self.dim):
            Jacs[k, :, k] = ps_marg[k] / ps_marg[k-1] * wls[k]

        # TODO: populate off-diagonal elements
        for k in range(1, self.dim):
            for j in range(k):
                grad_cond = (ps_grid_deriv[k][j] * ps_marg[k-1] - ps_grid[k] * ps_marg_deriv[k-1][j]) / ps_marg[k-1].square() * wls[k]
                Jacs[k, :, j] = self.oned_cdfs[k].eval_int_deriv(grad_cond, ls[:, k])
            
        return Jacs.reshape(self.dim, self.dim*n_ls)