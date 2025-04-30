from typing import Tuple

import torch
from torch import Tensor

from .abstract_irt import AbstractIRT
from ..ftt import Direction, TTFunc
from ..polynomials import CDF1D


class TTSIRT(AbstractIRT):
    r"""Squared inverse Rosenblatt transport.
    
    Parameters
    ----------
    potential:
        A function that receives an $n \times d$ matrix of samples and 
        returns an $n$-dimensional vector containing the potential 
        function of the target density evaluated at each sample.
    bases:
        An object containing information on the basis functions in each 
        dimension used during the FTT construction, and the mapping 
        between the approximation domain and the domain of the basis 
        functions.
    prev_approx: 
        A previously-constructed FTT object to use as a starting point 
        when constructing the FTT part of the TTSIRT. If passed in, the 
        bases and options associated with this approximation will be 
        inherited by the new TTSIRT, and the cores and interpolation 
        points will be used as a starting point for the new FTT.
    options:
        A set of options that control the construction of the FTT.
    input_data:
        An object that holds data used to construct and evaluate the 
        quality of the FTT approximation to the target function.
    tt_data:
        An object that holds information about the FTT, including the 
        cores and interpolation points.
    defensive:
        The defensive parameter, $\tau$, which ensures that the tails
        of the approximation are sufficiently heavy.

    References
    ----------
    Cui, T and Dolgov, S (2022). *[Deep composition of tensor-trains 
    using squared inverse Rosenblatt transports](https://doi.org/10.1007/s10208-021-09537-5).* 
    Foundations of Computational Mathematics, **22**, 1863--1922.

    """

    @property 
    def oned_cdfs(self) -> dict[int, CDF1D]:
        return self._oned_cdfs
    
    @oned_cdfs.setter 
    def oned_cdfs(self, value: dict) -> None:
        self._oned_cdfs = value
        return

    @property
    def approx(self) -> TTFunc:
        return self._approx

    @approx.setter 
    def approx(self, value: TTFunc) -> None:
        self._approx = value
        return

    @property 
    def defensive(self) -> Tensor:
        return self._defensive
    
    @defensive.setter
    def defensive(self, value: Tensor) -> None:
        self._defensive = value 
        return
    
    @property 
    def z(self) -> Tensor:
        return self._z 
    
    @z.setter
    def z(self, value: Tensor) -> None:
        self._z = value
        return

    @property 
    def z_func(self) -> Tensor:
        return self._z_func

    @z_func.setter
    def z_func(self, value: Tensor) -> None:
        self._z_func = value
        return

    def _marginalise_forward(self) -> None:
        """Computes each coefficient tensor required to evaluate the 
        marginal functions in each dimension, by iterating over the 
        dimensions of the approximation from last to first.
        """

        self.Rs_f[self.dim] = torch.tensor([[1.0]])
        polys = self.bases.polys
        cores = self.approx.tt_data.cores

        for k in range(self.dim-1, -1, -1):
            self.Bs_f[k] = torch.einsum("ijl, lk", cores[k], self.Rs_f[k+1])
            C_k = torch.einsum("ilk, lj", self.Bs_f[k], polys[k].mass_R)
            C_k = TTFunc.unfold_right(C_k)
            self.Rs_f[k] = torch.linalg.qr(C_k, mode="reduced")[1].T

        self.z_func = self.Rs_f[0].square().sum()
        self.z = self.z_func + self.defensive
        return 
    
    def _marginalise_backward(self) -> None:
        """Computes each coefficient tensor required to evaluate the 
        marginal functions in each dimension, by iterating over the 
        dimensions of the approximation from first to last.
        """
        
        self.Rs_b[-1] = torch.tensor([[1.0]])
        polys = self.bases.polys
        cores = self.approx.tt_data.cores

        for k in range(self.dim):
            self.Bs_b[k] = torch.einsum("il, ljk", self.Rs_b[k-1], cores[k])
            C_k = torch.einsum("jl, ilk", polys[k].mass_R, self.Bs_b[k])
            C_k = TTFunc.unfold_left(C_k)
            self.Rs_b[k] = torch.linalg.qr(C_k, mode="reduced")[1]

        self.z_func = self.Rs_b[self.dim-1].square().sum()
        self.z = self.z_func + self.defensive
        return

    def _eval_potential_local(self, ls: Tensor, direction: Direction) -> Tensor:

        dim_l = ls.shape[1]

        if direction == Direction.FORWARD:
            indices = torch.arange(dim_l)
            gs = self.approx._eval_local(ls, direction=direction)
            gs_sq = (gs @ self.Rs_f[dim_l]).square().sum(dim=1)
            
        else:
            i_min = self.dim - dim_l
            indices = torch.arange(self.dim-1, self.dim-dim_l-1, -1)
            gs = self.approx._eval_local(ls, direction=direction)
            gs_sq = (self.Rs_b[i_min-1] @ gs.T).square().sum(dim=0)
            
        # TODO: check that indices go backwards. This could be an issue 
        # if different bases are used in each dimension.
        neglogwls = self.bases.eval_measure_potential_local(ls, indices)
        neglogfls = self.z.log() - (gs_sq + self.defensive).log() + neglogwls
        return neglogfls

    def _eval_rt_local_forward(self, ls: Tensor) -> Tensor:

        n_ls, d_ls = ls.shape
        zs = torch.zeros_like(ls)
        Gs_prod = torch.ones((n_ls, 1))

        polys = self.bases.polys
        cores = self.approx.tt_data.cores
        Bs = self.Bs_f 
        cdfs = self.oned_cdfs
            
        for k in range(d_ls):
            
            # Compute (unnormalised) conditional PDF for each sample
            Ps = TTFunc.eval_core_213(polys[k], Bs[k], cdfs[k].nodes)
            gs = torch.einsum("jl, ilk -> ijk", Gs_prod, Ps)
            ps = gs.square().sum(dim=2) + self.defensive

            # Evaluate CDF to obtain corresponding uniform variates
            zs[:, k] = self.oned_cdfs[k].eval_cdf(ps, ls[:, k])

            # Compute incremental product of tensor cores for each sample
            Gs = TTFunc.eval_core_213(polys[k], cores[k], ls[:, k])
            Gs_prod = torch.einsum("il, ilk -> ik", Gs_prod, Gs)

        return zs
    
    def _eval_rt_local_backward(self, ls: Tensor) -> Tensor:

        n_ls, d_ls = ls.shape
        zs = torch.zeros_like(ls)
        d_min = self.dim - d_ls
        Gs_prod = torch.ones((1, n_ls))

        polys = self.bases.polys
        cores = self.approx.tt_data.cores
        Bs = self.Bs_b 
        cdfs = self.oned_cdfs

        for i, k in enumerate(range(self.dim-1, d_min-1, -1), start=1):

            # Compute (unnormalised) conditional PDF for each sample
            Ps = TTFunc.eval_core_213(polys[k], Bs[k], cdfs[k].nodes)
            gs = torch.einsum("ijl, lk -> ijk", Ps, Gs_prod)
            ps = gs.square().sum(dim=1) + self.defensive

            # Evaluate CDF to obtain corresponding uniform variates
            zs[:, -i] = self.oned_cdfs[k].eval_cdf(ps, ls[:, -i])
            
            # Compute incremental product of tensor cores for each sample
            Gs = TTFunc.eval_core_213(polys[k], cores[k], ls[:, -i])
            Gs_prod = torch.einsum("ijl, li -> ji", Gs, Gs_prod)

        return zs

    def _eval_rt_local(self, ls: Tensor, direction: Direction) -> Tensor:
        if direction == Direction.FORWARD:
            zs = self._eval_rt_local_forward(ls)
        else:
            zs = self._eval_rt_local_backward(ls)
        return zs

    def _eval_irt_local_forward(self, zs: Tensor) -> Tuple[Tensor, Tensor]:
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
        gs_sq:
            An n-dimensional vector containing the square of the FTT 
            approximation to the square root of the target function, 
            evaluated at each sample in zs.
        
        """

        n_zs, d_zs = zs.shape
        ls = torch.zeros_like(zs)
        gs = torch.ones((n_zs, 1))

        polys = self.bases.polys
        cores = self.approx.tt_data.cores
        Bs = self.Bs_f
        cdfs = self.oned_cdfs

        for k in range(d_zs):
            
            Ps = TTFunc.eval_core_213(polys[k], Bs[k], cdfs[k].nodes)
            gls = torch.einsum("jl, ilk", gs, Ps)
            ps = gls.square().sum(dim=2) + self.defensive
            ls[:, k] = self.oned_cdfs[k].invert_cdf(ps, zs[:, k])

            Gs = TTFunc.eval_core_213(polys[k], cores[k], ls[:, k])
            gs = torch.einsum("il, ilk -> ik", gs, Gs)
        
        gs_sq = (gs @ self.Rs_f[d_zs]).square().sum(dim=1)
        return ls, gs_sq
    
    def _eval_irt_local_backward(self, zs: Tensor) -> Tuple[Tensor, Tensor]:
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
        gs_sq:
            An n-dimensional vector containing the square of the FTT 
            approximation to the square root of the target function, 
            evaluated at each sample in zs.
        
        """

        n_zs, d_zs = zs.shape
        ls = torch.zeros_like(zs)
        gs = torch.ones((n_zs, 1))
        d_min = self.dim - d_zs

        polys = self.bases.polys
        cores = self.approx.tt_data.cores
        Bs = self.Bs_b
        cdfs = self.oned_cdfs
        
        for i, k in enumerate(range(self.dim-1, d_min-1, -1), start=1):

            Ps = TTFunc.eval_core_231(polys[k], Bs[k], cdfs[k].nodes)
            gls = torch.einsum("ilk, jl", Ps, gs)
            ps = gls.square().sum(dim=2) + self.defensive
            ls[:, -i] = self.oned_cdfs[k].invert_cdf(ps, zs[:, -i])

            Gs = TTFunc.eval_core_231(polys[k], cores[k], ls[:, -i])
            gs = torch.einsum("il, ilk -> ik", gs, Gs)

        gs_sq = (self.Rs_b[d_min-1] @ gs.T).square().sum(dim=0)
        return ls, gs_sq

    def _eval_irt_local(
        self, 
        zs: Tensor,
        direction: Direction
    ) -> Tuple[Tensor, Tensor]:

        if direction == Direction.FORWARD:
            ls, gs_sq = self._eval_irt_local_forward(zs)
        else:
            ls, gs_sq = self._eval_irt_local_backward(zs)
        
        indices = self._get_transform_indices(zs.shape[1], direction)
        
        neglogpls = -(gs_sq + self.defensive).log()
        neglogwls = self.bases.eval_measure_potential_local(ls, indices)
        neglogfls = self.z.log() + neglogpls + neglogwls

        return ls, neglogfls

    def _eval_cirt_local_forward(
        self, 
        ls_x: Tensor, 
        zs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        n_xs, d_xs = ls_x.shape
        n_zs, d_zs = zs.shape
        ls_y = torch.zeros_like(zs)

        polys = self.bases.polys
        cores = self.approx.tt_data.cores
        Bs = self.Bs_f
        cdfs = self.oned_cdfs
        
        Gs_prod = torch.ones((n_xs, 1, 1))

        for k in range(d_xs-1):
            Gs = TTFunc.eval_core_213(polys[k], cores[k], ls_x[:, k])
            Gs_prod = TTFunc.batch_mul(Gs_prod, Gs)
        
        k = d_xs-1

        Ps = TTFunc.eval_core_213(polys[k], Bs[k], ls_x[:, k])
        gs_marg = TTFunc.batch_mul(Gs_prod, Ps)
        ps_marg = gs_marg.square().sum(dim=(1, 2)) + self.defensive

        Gs = TTFunc.eval_core_213(polys[k], cores[k], ls_x[:, k])
        Gs_prod = TTFunc.batch_mul(Gs_prod, Gs)

        # Generate conditional samples
        for i, k in enumerate(range(d_xs, self.dim)):
            
            Ps = TTFunc.eval_core_213(polys[k], Bs[k], cdfs[k].nodes)
            gs = torch.einsum("mij, ljk -> lmk", Gs_prod, Ps)
            ps = gs.square().sum(dim=2) + self.defensive
            ls_y[:, i] = self.oned_cdfs[k].invert_cdf(ps, zs[:, i])

            Gs = TTFunc.eval_core_213(polys[k], cores[k], ls_y[:, i])
            Gs_prod = TTFunc.batch_mul(Gs_prod, Gs)

        ps = Gs_prod.flatten().square() + self.defensive

        indices = d_xs + torch.arange(d_zs)
        neglogwls_y = self.bases.eval_measure_potential_local(ls_y, indices)
        neglogfls_y = ps_marg.log() - ps.log() + neglogwls_y

        return ls_y, neglogfls_y
    
    def _eval_cirt_local_backward(
        self, 
        ls_x: Tensor, 
        zs: Tensor
    ) -> Tuple[Tensor, Tensor]:

        n_zs, d_zs = zs.shape
        ls_y = torch.zeros_like(zs)

        polys = self.bases.polys
        cores = self.approx.tt_data.cores
        Bs = self.Bs_b
        cdfs = self.oned_cdfs

        Gs_prod = torch.ones((n_zs, 1, 1))

        for i, k in enumerate(range(self.dim-1, d_zs, -1), start=1):
            Gs = TTFunc.eval_core_213(polys[k], cores[k], ls_x[:, -i])
            Gs_prod = TTFunc.batch_mul(Gs, Gs_prod)

        Ps = TTFunc.eval_core_213(polys[d_zs], Bs[d_zs], ls_x[:, 0])
        gs_marg = TTFunc.batch_mul(Ps, Gs_prod)
        ps_marg = gs_marg.square().sum(dim=(1, 2)) + self.defensive

        Gs = TTFunc.eval_core_213(polys[d_zs], cores[d_zs], ls_x[:, 0])
        Gs_prod = TTFunc.batch_mul(Gs, Gs_prod)

        # Generate conditional samples
        for k in range(d_zs-1, -1, -1):

            Ps = TTFunc.eval_core_213(polys[k], Bs[k], cdfs[k].nodes)
            gs = torch.einsum("lij, mjk -> lmi", Ps, Gs_prod)
            ps = gs.square().sum(dim=2) + self.defensive
            ls_y[:, k] = self.oned_cdfs[k].invert_cdf(ps, zs[:, k])

            Gs = TTFunc.eval_core_213(polys[k], cores[k], ls_y[:, k])
            Gs_prod = TTFunc.batch_mul(Gs, Gs_prod)

        ps = Gs_prod.flatten().square() + self.defensive

        indices = torch.arange(d_zs-1, -1, -1)
        neglogwls_y = self.bases.eval_measure_potential_local(ls_y, indices)
        neglogfls_y = ps_marg.log() - ps.log() + neglogwls_y

        return ls_y, neglogfls_y

    def _eval_cirt_local(
        self, 
        ls_x: Tensor, 
        zs: Tensor,
        direction: Direction
    ) -> Tuple[Tensor, Tensor]:

        if direction == Direction.FORWARD:
            ls_y, neglogfls_y = self._eval_cirt_local_forward(ls_x, zs)
        else:
            ls_y, neglogfls_y = self._eval_cirt_local_backward(ls_x, zs)

        return ls_y, neglogfls_y
    
    def _eval_potential_grad_local(self, ls: Tensor) -> Tensor:

        polys = self.bases.polys
        cores = self.approx.tt_data.cores

        zs = self._eval_rt_local_forward(ls)
        ls, gs_sq = self._eval_irt_local_forward(zs)
        n_ls = ls.shape[0]
        ps = gs_sq + self.defensive
        neglogws = self.bases.eval_measure_potential_local(ls)
        ws = torch.exp(-neglogws)
        fs = ps * ws  # Don't need to normalise as derivative ends up being a ratio
        
        Gs_prod = torch.ones((n_ls, 1, 1))
        
        dwdls = {k: torch.ones((n_ls, )) for k in range(self.dim)}
        dGdls = {k: torch.ones((n_ls, 1, 1)) for k in range(self.dim)}
        
        for k in range(self.dim):

            ws_k = polys[k].eval_measure(ls[:, k])
            dwdls_k = polys[k].eval_measure_deriv(ls[:, k])

            Gs_k = TTFunc.eval_core_213(polys[k], cores[k], ls[:, k])
            dGdls_k = TTFunc.eval_core_213_deriv(polys[k], cores[k], ls[:, k])
            Gs_prod = TTFunc.batch_mul(Gs_prod, Gs_k)
            
            for j in range(self.dim):
                if k == j:
                    dwdls[j] *= dwdls_k
                    dGdls[j] = TTFunc.batch_mul(dGdls[j], dGdls_k)
                else:
                    dwdls[j] *= ws_k
                    dGdls[j] = TTFunc.batch_mul(dGdls[j], Gs_k)
        
        dfdls = torch.zeros_like(ls)
        deriv = torch.zeros_like(ls)
        gs = Gs_prod.sum(dim=(1, 2)) 

        for k in range(self.dim):
            dGdls_k = dGdls[k].sum(dim=(1, 2))
            dfdls[:, k] = ps * dwdls[k] + 2.0 * gs * dGdls_k * ws
            deriv[:, k] = -dfdls[:, k] / fs

        return deriv

    def _eval_rt_jac_local_forward(self, ls: Tensor) -> Tensor:

        polys = self.bases.polys
        cores = self.approx.tt_data.cores
        Bs = self.Bs_f
        cdfs = self.oned_cdfs

        Gs: dict[int, Tensor] = {}
        Gs_deriv: dict[int, Tensor] = {}
        Ps: dict[int, Tensor] = {}
        Ps_deriv: dict[int, Tensor] = {}
        Ps_grid: dict[int, Tensor] = {}

        ps_marg: dict[int, Tensor] = {}
        ps_marg[-1] = self.z
        ps_marg_deriv: dict[int, Tensor] = {}
        ps_grid: dict[int, Tensor] = {}
        ps_grid_deriv: dict[int, Tensor] = {}
        wls: dict[int, Tensor] = {}

        n_ls = ls.shape[0]
        Jacs = torch.zeros((self.dim, n_ls, self.dim))

        Gs_prod = {} 
        Gs_prod[-1] = torch.ones((n_ls, 1, 1))

        for k in range(self.dim):

            # Evaluate weighting function
            wls[k] = polys[k].eval_measure(ls[:, k])

            # Evaluate kth tensor core and derivative
            Gs[k] = TTFunc.eval_core_213(polys[k], cores[k], ls[:, k])
            Gs_deriv[k] = TTFunc.eval_core_213_deriv(polys[k], cores[k], ls[:, k])
            Gs_prod[k] = TTFunc.batch_mul(Gs_prod[k-1], Gs[k])

            # Evaluate kth marginalisation core and derivative
            Ps[k] = TTFunc.eval_core_213(polys[k], Bs[k], ls[:, k])
            Ps_deriv[k] = TTFunc.eval_core_213_deriv(polys[k], Bs[k], ls[:, k])
            Ps_grid[k] = TTFunc.eval_core_213(polys[k], Bs[k], cdfs[k].nodes)

            # Evaluate marginal probability for the first k elements of 
            # each sample
            gs = TTFunc.batch_mul(Gs_prod[k-1], Ps[k])
            ps_marg[k] = gs.square().sum(dim=(1, 2)) + self.defensive

            # Compute (unnormalised) marginal PDF at CDF nodes for each sample
            gs_grid = torch.einsum("mij, ljk -> lmik", Gs_prod[k-1], Ps_grid[k])
            ps_grid[k] = gs_grid.square().sum(dim=(2, 3)) + self.defensive

        # Derivatives of marginal PDF
        for k in range(self.dim-1):
            ps_marg_deriv[k] = {}
            
            for j in range(k+1):

                prod = TTFunc.batch_mul(Gs_prod[k-1], Ps[k])
                prod_deriv = torch.ones((n_ls, 1, 1))

                for k_i in range(k):
                    core = Gs_deriv[k_i] if k_i == j else Gs[k_i]
                    prod_deriv = TTFunc.batch_mul(prod_deriv, core)
                core = Ps_deriv[k] if k == j else Ps[k]
                prod_deriv = TTFunc.batch_mul(prod_deriv, core)

                ps_marg_deriv[k][j] = 2 * (prod * prod_deriv).sum(dim=(1, 2))

        for k in range(1, self.dim):
            ps_grid_deriv[k] = {}

            for j in range(k):

                prod = torch.einsum("mij, ljk -> lmik", Gs_prod[k-1], Ps_grid[k])
                prod_deriv = torch.ones((n_ls, 1, 1))

                for k_i in range(k):
                    core = Gs_deriv[k_i] if k_i == j else Gs[k_i]
                    prod_deriv = TTFunc.batch_mul(prod_deriv, core)
                prod_deriv = torch.einsum("mij, ljk -> lmik", prod_deriv, Ps_grid[k])
                
                ps_grid_deriv[k][j] = 2 * (prod * prod_deriv).sum(dim=(2, 3))

        # Populate diagonal elements
        for k in range(self.dim):
            Jacs[k, :, k] = ps_marg[k] / ps_marg[k-1] * wls[k]

        # Populate off-diagonal elements
        for k in range(1, self.dim):
            for j in range(k):
                grad_cond = (ps_grid_deriv[k][j] * ps_marg[k-1] 
                             - ps_grid[k] * ps_marg_deriv[k-1][j]) / ps_marg[k-1].square()
                if polys[k].constant_weight:
                    grad_cond *= wls[k]
                Jacs[k, :, j] = self.oned_cdfs[k].eval_int_deriv(grad_cond, ls[:, k])

        return Jacs
    
    def _eval_rt_jac_local_backward(self, ls: Tensor) -> Tensor:

        polys = self.bases.polys
        cores = self.approx.tt_data.cores
        Bs = self.Bs_b
        cdfs = self.oned_cdfs

        Gs: dict[int, Tensor] = {}
        Gs_deriv: dict[int, Tensor] = {}
        Ps: dict[int, Tensor] = {}
        Ps_deriv: dict[int, Tensor] = {}
        Ps_grid: dict[int, Tensor] = {}

        ps_marg: dict[int, Tensor] = {}
        ps_marg[self.dim] = self.z
        ps_marg_deriv: dict[int, Tensor] = {}
        ps_grid: dict[int, Tensor] = {}
        ps_grid_deriv: dict[int, Tensor] = {}
        wls: dict[int, Tensor] = {}

        n_ls = ls.shape[0]
        Jacs = torch.zeros((self.dim, n_ls, self.dim))

        Gs_prod = {} 
        Gs_prod[self.dim] = torch.ones((n_ls, 1, 1))

        for k in range(self.dim-1, -1, -1):

            # Evaluate weighting function
            wls[k] = polys[k].eval_measure(ls[:, k])

            # Evaluate kth tensor core and derivative
            Gs[k] = TTFunc.eval_core_231(polys[k], cores[k], ls[:, k])
            Gs_deriv[k] = TTFunc.eval_core_231_deriv(polys[k], cores[k], ls[:, k])
            Gs_prod[k] = TTFunc.batch_mul(Gs_prod[k+1], Gs[k])

            # Evaluate kth marginalisation core and derivative
            Ps[k] = TTFunc.eval_core_231(polys[k], Bs[k], ls[:, k])
            Ps_deriv[k] = TTFunc.eval_core_231_deriv(polys[k], Bs[k], ls[:, k])
            Ps_grid[k] = TTFunc.eval_core_231(polys[k], Bs[k], cdfs[k].nodes)

            # Evaluate marginal probability for the first k elements of 
            # each sample
            gs = TTFunc.batch_mul(Gs_prod[k+1], Ps[k])
            ps_marg[k] = gs.square().sum(dim=(1, 2)) + self.defensive

            # Compute (unnormalised) marginal PDF at CDF nodes for each sample
            gs_grid = torch.einsum("mij, ljk -> lmik", Gs_prod[k+1], Ps_grid[k])
            ps_grid[k] = gs_grid.square().sum(dim=(2, 3)) + self.defensive

        # Derivatives of marginal PDF
        for k in range(1, self.dim):
            ps_marg_deriv[k] = {}

            for j in range(k, self.dim):

                prod = TTFunc.batch_mul(Gs_prod[k+1], Ps[k])
                prod_deriv = torch.ones((n_ls, 1, 1))

                for k_i in range(self.dim-1, k, -1):
                    core = Gs_deriv[k_i] if k_i == j else Gs[k_i]
                    prod_deriv = TTFunc.batch_mul(prod_deriv, core)
                core = Ps_deriv[k] if k == j else Ps[k] 
                prod_deriv = TTFunc.batch_mul(prod_deriv, core)

                ps_marg_deriv[k][j] = 2 * (prod * prod_deriv).sum(dim=(1, 2))

        for k in range(self.dim-1):
            ps_grid_deriv[k] = {}

            for j in range(k+1, self.dim):

                prod = torch.einsum("mij, ljk -> lmik", Gs_prod[k+1], Ps_grid[k])
                prod_deriv = torch.ones((n_ls, 1, 1))

                for k_i in range(self.dim-1, k, -1):
                    core = Gs_deriv[k_i] if k_i == j else Gs[k_i]
                    prod_deriv = TTFunc.batch_mul(prod_deriv, core)
                prod_deriv = torch.einsum("mij, ljk -> lmik", prod_deriv, Ps_grid[k])
                
                ps_grid_deriv[k][j] = 2 * (prod * prod_deriv).sum(dim=(2, 3))

        # Populate diagonal elements
        for k in range(self.dim):
            Jacs[k, :, k] = ps_marg[k] / ps_marg[k+1] * wls[k]

        # Populate off-diagonal elements
        for k in range(self.dim-1):
            for j in range(k+1, self.dim):
                grad_cond = (ps_grid_deriv[k][j] * ps_marg[k+1] 
                             - ps_grid[k] * ps_marg_deriv[k+1][j]) / ps_marg[k+1].square()
                if polys[k].constant_weight:
                    grad_cond *= wls[k]
                Jacs[k, :, j] = self.oned_cdfs[k].eval_int_deriv(grad_cond, ls[:, k])
            
        return Jacs

    def _eval_rt_jac_local(self, ls: Tensor, direction: Direction) -> Tensor:

        if direction == Direction.FORWARD:
            J = self._eval_rt_jac_local_forward(ls)
        else:
            J = self._eval_rt_jac_local_backward(ls)
        return J