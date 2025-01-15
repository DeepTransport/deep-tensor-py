from typing import Callable, Tuple

import torch

from .abstract_irt import AbstractIRT
# from .sirt import SIRT
from ..approx_bases import ApproxBases
from ..approx_func import ApproxFunc
from ..directions import Direction
from ..input_data import InputData
from ..options import TTOptions
from ..polynomials import Basis1D, CDF1D, construct_cdf
from ..tt_data import TTData
from ..tt_func import TTFunc
from ..tools import reshape_matlab


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
        
        # Define coefficient tensors and marginalisation coefficents
        self.Bs: dict[int, torch.Tensor] = {}
        self.Rs: dict[int, torch.Tensor] = {} 

        AbstractIRT.__init__(
            self,
            potential, 
            bases,
            approx, 
            options, 
            input_data,
            tt_data
        )
        
        self._int_dir = Direction.FORWARD # TEMP??
        self._order = None
        self._tau = tau

        def target_func(ls: torch.Tensor) -> torch.Tensor:
            """Returns the square root of the ratio between the target 
            density and the weighting function evaluated at a set of 
            points in the local domain ([-1, 1]^d).
            """
            return self.potential2density(potential, ls)

        self._approx = self.build_approximation(
            target_func, 
            bases,
            options, 
            input_data,
            tt_data
        )

        self._oned_cdfs = {}
        for k in range(self.bases.dim):
            self._oned_cdfs[k] = construct_cdf(
                poly=self.approx.bases.polys[k], 
                error_tol=self.approx.options.cdf_tol
            )

        self.marginalise()

        # TODO: figure out what this is for. I think this is set in the 
        # marginalise() function--so I'm not sure what's going on here.
        self.order = None 
        return

    @property 
    def oned_cdfs(self) -> dict[int, CDF1D]:
        return self._oned_cdfs

    @property
    def approx(self) -> ApproxFunc:
        return self._approx

    @approx.setter 
    def approx(self, value: ApproxFunc):
        self._approx = value

    @property
    def int_dir(self) -> Direction:
        return self._int_dir
    
    @property
    def order(self) -> torch.Tensor:
        return self._order
    
    @order.setter 
    def order(self, value: torch.Tensor):
        self._order = value
        return

    @property 
    def tau(self) -> torch.Tensor:
        return self._tau
    
    @property 
    def z(self) -> torch.Tensor:
        return self._z 
    
    @property 
    def z_func(self) -> torch.Tensor:
        return self._z_func

    def _marginalise_forward(self) -> None:
        """TODO: write docstring."""

        self.order = torch.arange(self.bases.dim)
        self.Rs[self.bases.dim] = torch.tensor([[1.0]])

        for k in range(self.bases.dim-1, -1, -1):
            
            poly_k = self.approx.bases.polys[k]
            A_k = self.approx.data.cores[k]
            r_p, n_k, r_k = A_k.shape

            B_k = A_k.swapdims(0, 2).reshape(r_k, r_p * n_k).T @ self.Rs[k+1]
            self.Bs[k] = B_k.T.reshape(r_k, n_k, r_p).swapdims(0, 2)
            B_k = self.Bs[k].swapdims(2, 1).reshape(r_p * r_k, n_k).T
            C_k = poly_k.mass_r(B_k).T.reshape(r_p, n_k * r_k).T
            self.Rs[k] = torch.linalg.qr(C_k, mode="reduced")[1].T

        self._z_func = self.Rs[0].square().sum()
        return 
    
    def _marginalise_backward(self) -> None:
        """TODO: write docstring."""
        
        self.order = torch.arange(self.bases.dim-1, -1, -1)
        self.Rs[-1] = torch.tensor([[1.0]])

        for k in range(self.bases.dim):
            
            poly_k = self.approx.bases.polys[k]
            A_k = self.approx.data.cores[k]
            r_p, n_k, r_k = A_k.shape

            B_k = self.Rs[k-1] @ A_k.swapdims(0, 2).reshape(n_k * r_k, r_p).T
            self.Bs[k] = B_k.T.reshape(r_k, n_k, r_p).swapdims(0, 2)
            B_k = self.Bs[k].permute(2, 0, 1).reshape(r_p * r_k, n_k).T
            C_k = poly_k.mass_r(B_k).T.reshape(r_k, n_k * r_p).T
            self.Rs[k] = torch.linalg.qr(C_k, mode="reduced")[1]

        self._z_func = self.Rs[self.bases.dim-1].square().sum()
        return

    def _eval_irt_local_nograd_forward(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the inverse Rosenblatt transport by iterating over
        the dimensions from first to last.
        """

        num_z, dim_z = zs.shape
        ls = torch.zeros_like(zs)
        fls_leq = torch.ones((num_z, 1))

        for k in range(dim_z):

            rank_p = self.approx.data.cores[k].shape[0]
            rank_k = self.approx.data.cores[k].shape[-1]
            num_nodes = self.oned_cdfs[k].cardinality

            # Evaluate the current core at each node of the current CDF
            G_ks = self.eval_oned_core_213(
                self.approx.bases.polys[k], 
                self.Bs[k], 
                self.oned_cdfs[k].nodes
            ).T.reshape(rank_k * num_nodes, rank_p).T

            gls_sq = ((fls_leq @ G_ks).T
                      .reshape(-1, num_z * num_nodes).T
                      .square()
                      .sum(dim=1)
                      .reshape(num_nodes, num_z))

            ls[:, k] = self.oned_cdfs[k].invert_cdf(
                gls_sq + self.tau, 
                zs[:, k]
            )

            T2 = self.eval_oned_core_213(
                self.approx.bases.polys[k], 
                self.approx.data.cores[k], 
                ls[:, k]
            )

            ii = torch.arange(num_z).repeat(rank_p)
            jj = (torch.arange(rank_p * num_z)
                  .reshape(num_z, rank_p).T
                  .flatten())

            indices = torch.vstack((ii[None, :], jj[None, :]))
            values = fls_leq.T.flatten()
            size = (num_z, rank_p * num_z)
            B = torch.sparse_coo_tensor(indices, values, size)
            
            fls_leq = B @ T2
            fls_leq[fls_leq.isnan()] = 0.0
        
        pi_ls = (fls_leq @ self.Rs[dim_z]).square().sum(dim=1)

        return ls, pi_ls
    
    def _eval_irt_local_nograd_backward(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """TODO: write docstring"""
        
        num_z, dim_z = zs.shape
        rs = torch.zeros_like(zs)

        frg = torch.ones((1, num_z))
        i_min = self.bases.dim - dim_z
        
        for k in range(self.bases.dim-1, i_min-1, -1):
            
            k_ind = k - i_min
            rank_p = self.approx.data.cores[k].shape[0]
            rank_k = self.approx.data.cores[k].shape[-1]
            num_nodes = self.oned_cdfs[k].cardinality

            T1 = self.eval_oned_core_213(
                self.approx.bases.polys[k], 
                self.Bs[k],
                self.oned_cdfs[k].nodes
            ).T.reshape(rank_k, rank_p * num_nodes).T

            # Compute the value of the conditional pdf at each sample
            pdf_vals = ((T1 @ frg).T.reshape(num_nodes * num_z, -1).T
                                    .square()
                                    .sum(dim=0)
                                    .reshape(num_z, num_nodes).T)

            rs[:, k_ind] = self.oned_cdfs[k].invert_cdf(
                pdf_vals + self.tau, 
                zs[:, k_ind]
            )

            T2 = self.eval_oned_core_231(
                self.approx.bases.polys[k], 
                self.approx.data.cores[k],
                rs[:, k_ind]
            )

            ii = torch.arange(rank_k * num_z)
            jj = torch.arange(num_z).repeat_interleave(rank_k)
            indices = torch.vstack((ii[None, :], jj[None, :]))
            values = frg.T.flatten()
            size = (rank_k * num_z, num_z)
            
            B = torch.sparse_coo_tensor(indices, values, size)
            frg = T2.T @ B 
            frg[torch.isnan(frg)] = 0.0

        if dim_z < self.approx.bases.dim:
            frs = (self.Rs[i_min-1] @ frg).square().sum(dim=0)
        else:
            frs = frg.flatten().square()

        return rs, frs

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
            
        TODO: check this (and eval_measure_potential) with TC.

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

        if approx.use_amen:
            approx.round()  # TODO: write this

        return approx

    def marginalise(
        self, 
        direction: Direction=Direction.FORWARD
    ) -> None:
        """Computes each coefficient tensor (B_k) required to evaluate 
        the marginal functions in each dimension, as well as the 
        normalising constant, z. 

        Parameters
        ----------
        TODO

        Returns
        -------
        None

        Notes
        -----
        Updates self.Bs, self.z_func, self.z.

        """

        self._int_dir = direction
        
        if self._int_dir == Direction.FORWARD:
            self._marginalise_forward()
        else:
            self._marginalise_backward()
        
        self._z = self.z_func + self.tau
        return
    
    def eval_oned_core_213(
        self, 
        poly_k: Basis1D, 
        A_k: torch.Tensor, 
        xs: torch.Tensor 
    ) -> torch.Tensor:
        """Evaluates the kth tensor core at a given set of x values.

        Parameters
        ----------
        poly_k:
            The basis functions associated with the current dimension.
        A_k:
            The coefficient tensor associated with the current core.
        xs: 
            A vector of points at which to evaluate the current core.

        Returns
        -------
        G_k:
            A matrix of dimension r_{k-1}n_{k} * r_{k}, corresponding 
            to evaluations of the kth core at each value of xs stacked 
            on top of one another.
        
        """
        
        r_p, n_k, r_k = A_k.shape
        n_x = xs.numel()

        coeffs = A_k.permute(2, 0, 1).reshape(r_p * r_k, n_k).T
        G_k = (poly_k.eval_radon(coeffs, xs).T
               .reshape(r_k, r_p, n_x)
               .swapdims(1, 2)
               .reshape(r_k, r_p * n_x).T)
        return G_k
        
    def eval_oned_core_213_deriv(
        self, 
        poly: Basis1D, 
        core: torch.Tensor, 
        xs: torch.Tensor 
    ) -> torch.Tensor:

        raise NotImplementedError()

    def eval_oned_core_231(
        self, 
        poly: Basis1D, 
        A_k: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the kth tensor core at a given set of x values.

        Parameters
        ----------
        poly_k:
            The basis functions associated with the current dimension.
        A_k:
            The coefficient tensor associated with the current core.
        xs: 
            A vector of points at which to evaluate the current core.

        Returns
        -------
        # G_k:
        #     A matrix of dimension r_{k}n_{k} * r_{k-1}, corresponding 
        #     to evaluations of the kth core at each value of xs stacked 
        #     on top of one another.
        """
        
        r_p, n_k, r_k = A_k.shape
        n_x = x.numel()

        coeffs = A_k.swapdims(1, 2).reshape(r_p * r_k, n_k).T
        G_k = (poly.eval_radon(coeffs, x).T
               .reshape(r_p, r_k, n_x)
               .swapdims(1, 2)
               .reshape(r_p, r_k * n_x).T)
        return G_k
    
    def eval_oned_core_231_deriv(
        self, 
        poly: Basis1D,
        core: torch.Tensor,
        xs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""

        raise NotImplementedError()
    
    def eval_potential_local(
        self, 
        ls: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the normalised (marginal) PDF represented by the 
        squared FTT.
        
        TODO: finish docstring.
        
        """

        dim_z = ls.shape[1]

        # d = self.approx.dim
        if self.int_dir == Direction.FORWARD:

            fxl = self.approx.eval_block(ls, direction=self.int_dir)
            
            # fx = (fxl @ self.Rs[dim_z]).square().sum(dim=1)

            if dim_z < self.approx.dim:
                fx = (fxl @ self.Rs[dim_z]).square().sum(dim=1)
            else: 
                fx = fxl.square()

            indices = torch.arange(dim_z)
            
        else:

            fxg = self.approx.eval_block(ls, direction=self.int_dir) # TODO: fix the int_dir thing

            i_min = self.approx.dim - dim_z
            # fx = (self.Rs[i_min-1] @ fxg).square().sum(dim=0)

            if dim_z < self.approx.dim:
                fx = (self.Rs[i_min-1] @ fxg).square().sum(dim=0)
            else:
                fx = fxg.square()

            indices = torch.arange(self.bases.dim-1, self.bases.dim-dim_z-1, -1)
            
        neglogwls = self.approx.bases.eval_measure_potential_local(ls, indices)  # TODO: check that indices go backwards
        neglogfls = self.z.log() - (fx + self.tau).log() + neglogwls

        return neglogfls

    def eval_rt_jac_local(
        self, 
        xs: torch.Tensor, 
        zs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()
    
    def eval_rt_local(
        self, 
        ls: torch.Tensor
    ) -> torch.Tensor:

        num_r, dim_r = ls.shape
        zs = torch.zeros_like(ls)

        if self.int_dir == Direction.FORWARD:

            frl = torch.ones((num_r, 1))
            
            for k in range(dim_r):
                
                rank_p = self.approx.data.cores[k].shape[0]
                rank_k = self.approx.data.cores[k].shape[-1]
                num_nodes = self.oned_cdfs[k].cardinality

                T1 = self.eval_oned_core_213(
                    self.bases.polys[k],
                    self.Bs[k],
                    self.oned_cdfs[k].nodes
                ).T.reshape(rank_k * num_nodes, rank_p).T

                # Squared TT
                pdf_vals = ((frl @ T1).T.reshape(-1, num_r * num_nodes).T 
                                      .square()
                                      .sum(dim=1)
                                      .reshape(num_nodes, num_r))
                
                zs[:, k] = self.oned_cdfs[k].eval_cdf(pdf_vals+self.tau, ls[:, k])

                # Evaluate the updated basis function
                T2 = self.eval_oned_core_213(
                    self.bases.polys[k], 
                    self.approx.data.cores[k], 
                    ls[:, k]
                )

                ii = torch.arange(num_r).repeat(rank_p)
                jj = (torch.arange(rank_p * num_r)
                           .reshape(num_r, rank_p).T
                           .flatten())

                indices = torch.vstack((ii[None, :], jj[None, :]))
                values = frl.T.flatten()
                size = (num_r, rank_p * num_r)

                B = torch.sparse_coo_tensor(indices, values, size)
                frl = B @ T2
        
        else:
            
            # TODO: check this

            k_min = self.approx.dim - dim_r
            frg = torch.ones((1, num_r))

            for k in range(self.approx.dim-1, k_min-1, -1):
                
                k_ind = k - k_min
                rank_k = self.approx.data.cores[k].shape[-1]
                num_nodes = self.oned_cdfs[k].cardinality

                T1 = self.eval_oned_core_213(
                    self.bases.polys[k],
                    self.Bs[k],
                    self.oned_cdfs[k].nodes
                )
                T1 = reshape_matlab(T1, (-1, rank_k))

                pk = reshape_matlab(T1 @ frg, (-1, num_nodes * num_r))
                pk = reshape_matlab(pk.square().sum(dim=0), (num_nodes, num_r))

                zs[:, k_ind] = self.oned_cdfs[k].eval_cdf(pk + self.tau, ls[:, k_ind])

                T2 = self.eval_oned_core_231(
                    self.bases.polys[k], 
                    self.approx.data.cores[k],
                    ls[:, k_ind]
                )

                ii = torch.arange(rank_k * num_r)
                jj = torch.arange(num_r).repeat_interleave(rank_k)

                indices = torch.vstack((ii[None, :], jj[None, :]))
                values = frg.T.flatten()
                size = (rank_k * num_r, num_r)
                
                B = torch.sparse_coo_tensor(indices, values, size)
                frg = T2.T @ B

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

        if dim_z != self.approx.bases.dim:
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

                T1 = self.eval_oned_core_213(
                    self.approx.bases.polys[k], 
                    self.Bs[k], 
                    self.oned_cdfs[k].nodes
                )
                T1 = reshape_matlab(T1, (rkm, -1))

                pk = reshape_matlab(frl @ T1, (num_z * num_nodes, -1)) ** 2
                pk = reshape_matlab(torch.sum(pk, dim=1), (num_z, num_nodes))

                rs[:, k] = self.oned_cdfs[k].invert_cdf(pk + self.tau, zs[:, k])

                T2 = self.eval_oned_core_213(
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
    
    def eval_irt_local_nograd(
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
            ls, gls_sq = self._eval_irt_local_nograd_forward(zs)
        else:
            ls, gls_sq = self._eval_irt_local_nograd_backward(zs)
        
        indices = self.get_transform_indices(zs.shape[1])
        
        neglogpls = -(gls_sq + self.tau).log()
        neglogwls = self.bases.eval_measure_potential_local(ls, indices)
        neglogfls = self.z.log() + neglogpls + neglogwls

        return ls, neglogfls

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
            
            T2 = self.eval_oned_core_213(
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

        T2 = self.eval_oned_core_213(
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
        
        T2 = self.eval_oned_core_213(
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

            T1 = self.eval_oned_core_213(
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

            T2 = self.eval_oned_core_213(
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
            
            T2 = self.eval_oned_core_231(
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

        T2 = self.eval_oned_core_231(
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
        
        T2 = self.eval_oned_core_231(
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
                self.eval_oned_core_213(
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

            T2 = self.eval_oned_core_231(
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