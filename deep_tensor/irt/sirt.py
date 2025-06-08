from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor
from torch.autograd.functional import jacobian
from torch.quasirandom import SobolEngine

from ..ftt import (
    ApproxBases, Direction, InputData, TTData, 
    AbstractTTFunc, TTFunc, SavedTTFunc
)
from ..options import TTOptions
from ..polynomials import Basis1D, CDF1D, construct_cdf
from ..preconditioners.preconditioner import Preconditioner
from ..tools import check_finite


PotentialFunc = Callable[[Tensor], Tensor]

SUBSET2DIRECTION = {
    "first": Direction.FORWARD,
    "last": Direction.BACKWARD
}


class AbstractSIRT():

    @property
    def input_data(self) -> InputData:
        return self._input_data 
    
    @input_data.setter 
    def input_data(self, value: InputData) -> None:
        self._input_data = value 
        return

    @property 
    def bases(self) -> ApproxBases:
        return self._bases 
    
    @bases.setter 
    def bases(self, value: ApproxBases) -> None:
        self._bases = value 
        return
    
    @property 
    def oned_cdfs(self) -> Dict[int, CDF1D]:
        return self._oned_cdfs
    
    @oned_cdfs.setter 
    def oned_cdfs(self, value: Dict[int, CDF1D]) -> None:
        self._oned_cdfs = value 
        return
    
    @property 
    def approx(self) -> AbstractTTFunc:
        return self._approx
    
    @approx.setter 
    def approx(self, value: AbstractTTFunc) -> None:
        self._approx = value 
        return
    
    @property 
    def defensive(self) -> float:
        return self._defensive
    
    @defensive.setter 
    def defensive(self, value: float) -> None:
        self._defensive = value 
        return
    
    @property 
    def z_func(self) -> Tensor:
        return self._z_func
    
    @z_func.setter 
    def z_func(self, value: Tensor) -> None:
        self._z_func = value 
        return 
    
    @property 
    def dim(self) -> int:
        return self.bases.dim 
    
    @property
    def z(self) -> Tensor:
        return self.defensive + self.z_func
    
    @property 
    def Bs_f(self) -> Dict[int, Tensor]:
        return self._Bs_f
    
    @Bs_f.setter 
    def Bs_f(self, value: Dict[int, Tensor]) -> None:
        self._Bs_f = value 
        return
    
    @property 
    def Bs_b(self) -> Dict[int, Tensor]:
        return self._Bs_b
    
    @Bs_b.setter 
    def Bs_b(self, value: Dict[int, Tensor]) -> None:
        self._Bs_b = value 
        return
    
    @property 
    def Rs_f(self) -> Dict[int, Tensor]:
        return self._Rs_f
    
    @Rs_f.setter 
    def Rs_f(self, value: Dict[int, Tensor]) -> None:
        self._Rs_f = value 
        return
    
    @property 
    def Rs_b(self) -> Dict[int, Tensor]:
        return self._Rs_b
    
    @Rs_b.setter 
    def Rs_b(self, value: Dict[int, Tensor]) -> None:
        self._Rs_b = value 
        return

    def _construct_cdfs(self, tol: float) -> Dict[int, CDF1D]:
        oned_cdfs = {}
        for k in range(self.dim):
            oned_cdfs[k] = construct_cdf(self.bases.polys[k], error_tol=tol)
        return oned_cdfs

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
        """Evaluates the Rosenblatt transport Z = R(L), where L is the 
        target random variable mapped into the local domain, and Z is 
        uniform.

        Parameters
        ----------
        ls:
            An n * d matrix containing samples from the local domain.
        direction:
            The direction in which to iterate over the tensor cores.
        
        Returns
        -------
        zs:
            An n * d matrix containing the result of applying the 
            inverse Rosenblatt transport to each sample in ls.
        
        """
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
        """Converts a set of realisations of a standard uniform 
        random variable, Z, to the corresponding realisations of the 
        local target random variable, by applying the inverse 
        Rosenblatt transport.
        
        Parameters
        ----------
        zs: 
            An n * d matrix containing values on [0, 1]^d.
        direction:
            The direction in which to iterate over the tensor cores.

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
        """Evaluates the inverse of the conditional squared Rosenblatt 
        transport.
        
        Parameters
        ----------
        ls_x:
            An n * m matrix containing samples from the local domain.
        zs:
            An n * (d-m) matrix containing samples from [0, 1]^{d-m},
            where m is the the dimension of the joint distribution of 
            X and Y.
        direction:
            The direction in which to iterate over the tensor cores.
        
        Returns
        -------
        ys:
            An n * (d-m) matrix containing the realisations of Y 
            corresponding to the values of zs after applying the 
            conditional inverse Rosenblatt transport.
        neglogfys:
            An n-dimensional vector containing the potential function 
            of the approximation to the conditional density of Y|X 
            evaluated at each sample in ys.
    
        """

        if direction == Direction.FORWARD:
            ls_y, neglogfls_y = self._eval_cirt_local_forward(ls_x, zs)
        else:
            ls_y, neglogfls_y = self._eval_cirt_local_backward(ls_x, zs)

        return ls_y, neglogfls_y
    
    def _eval_potential_grad_local(self, ls: Tensor) -> Tensor:
        """Evaluates the gradient of the potential function.
        
        Parameters
        ----------
        ls:
            An n * d set of samples from the local domain.
        
        Returns 
        -------
        grads:
            An n * d matrix containing the gradient of the potential 
            function at each element in ls.
        
        """

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
        ps_marg_deriv: dict[int, Dict] = {}
        ps_grid: dict[int, Tensor] = {}
        ps_grid_deriv: dict[int, Dict] = {}
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
        """Evaluates the Jacobian of the Rosenblatt transport.
        
        Parameters
        ----------
        zs: 
            An n * d matrix corresponding to evaluations of the 
            Rosenblatt transport at each sample in ls.
        direction:
            The direction in which to iterate over the tensor cores.
        
        Returns
        -------
        Js:
            A d * (d*n) matrix, where each d * d block contains the 
            Jacobian of the Rosenblatt transport evaluated at a given 
            sample: that is, J_ij = dz_i / dl_i.

        """
        if direction == Direction.FORWARD:
            J = self._eval_rt_jac_local_forward(ls)
        else:
            J = self._eval_rt_jac_local_backward(ls)
        return J
    
    def _get_transform_indices(self, dim_z: int, direction: Direction) -> Tensor:
        """TODO: write docstring."""

        if direction == Direction.FORWARD:
            return torch.arange(dim_z)
        elif direction == Direction.BACKWARD:
            return torch.arange(self.dim-dim_z, self.dim)

    def _eval_potential_grad_autodiff(self, xs: Tensor, subset: str | None) -> Tensor:
        """Evaluates the gradient of the potential using autodiff."""

        xs_shape = xs.shape

        def _eval_potential(xs: Tensor) -> Tensor:
            xs = xs.reshape(*xs_shape)
            return self.eval_potential(xs, subset).sum(dim=0)
        
        derivs = jacobian(_eval_potential, xs.flatten(), vectorize=True)
        return derivs.reshape(*xs_shape)

    def _eval_rt_jac_autodiff(self, xs: Tensor, subset: str) -> Tensor:
        """Evaluates the gradient of the Rosenblatt transport using 
        autodiff.
        """

        n_xs, d_xs = xs.shape

        def _eval_rt(xs: Tensor) -> Tensor:
            xs = xs.reshape(n_xs, d_xs)
            return self.eval_rt(xs, subset).sum(dim=0)
        
        Js = jacobian(_eval_rt, xs.flatten(), vectorize=True)
        return Js.reshape(d_xs, n_xs, d_xs)
    
    def round(self, tol: float | Tensor | None = None) -> None:
        """Rounds the TT cores. 
        
        Applies double rounding to get back to the starting direction.

        Parameters
        ----------
        tol:
            The tolerance to use when applying truncated SVD to round 
            each core. If `None`, will use `self.options.local_tol`.
        
        """
        self.approx._round(tol)
        return
    
    def _eval_potential_local(self, ls: Tensor, direction: Direction) -> Tensor:
        """Evaluates the normalised (marginal) PDF represented by the 
        squared FTT.
        
        Parameters
        ----------
        ls:
            An n * d matrix containing a set of samples from the local 
            domain.
        direction:
            The direction in which to iterate over the tensor cores.

        Returns
        -------
        neglogfls:
            An n-dimensional vector containing the approximation to the 
            target density function (transformed into the local domain) 
            at each element in ls.
        
        """

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
    
    def _eval_potential(self, xs: Tensor, subset: str) -> Tensor:
        r"""Evaluates the potential function.

        Returns the joint potential function, or the marginal potential 
        function for the first $k$ variables or the last $k$ variables,
        evaluated at a set of samples.

        Parameters
        ----------
        xs:
            An $n \times k$ matrix (where $1 \leq k \leq d$) containing 
            samples from the approximation domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        
        Returns
        -------
        neglogfxs:
            The potential function of the approximation to the target 
            density evaluated at each sample in `xs`.

        """
        direction = SUBSET2DIRECTION[subset]
        indices = self._get_transform_indices(xs.shape[1], direction)
        ls, dldxs = self.bases.approx2local(xs, indices)
        neglogfls = self._eval_potential_local(ls, direction)
        neglogfxs = neglogfls - dldxs.log().sum(dim=1)
        return neglogfxs
    
    def eval_potential(self, ms: Tensor, subset: str|None = None) -> Tensor:
        r"""Evaluates the potential function.

        Returns the joint potential function, or the marginal potential 
        function for the first $k$ variables or the last $k$ variables,
        evaluated at a set of samples.

        Parameters
        ----------
        ms:
            An $n \times k$ matrix (where $1 \leq k \leq d$) containing 
            samples from the approximation domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        
        Returns
        -------
        neglogfxs:
            The potential function of the approximation to the target 
            density evaluated at each sample in `xs`.

        """
        xs = self.preconditioner.Q_inv(ms, subset)
        neglogfxs = self._eval_potential(xs, subset)
        neglogabsdet_ms = self.preconditioner.neglogdet_Q_inv(ms, subset)
        neglogfms = neglogfxs + neglogabsdet_ms
        return neglogfms

    def _eval_pdf(self, xs: Tensor, subset: str | None = None) -> Tensor: 
        r"""Evaluates the density function.

        Returns the joint density function, or the marginal density 
        function for the first $k$ variables or the last $k$ variables, 
        evaluated at a set of samples.
        
        Parameters
        ----------
        xs:
            An $n \times k$ matrix (where $1 \leq k \leq d$) containing 
            samples from the reference domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        fxs:
            An $n$-dimensional vector containing the value of the 
            approximation to the target density evaluated at each 
            element in `xs`.
        
        """
        neglogfxs = self._eval_potential(xs, subset)
        fxs = torch.exp(-neglogfxs)
        return fxs

    def eval_pdf(self, ms: Tensor, subset: str | None = None) -> Tensor: 
        r"""Evaluates the density function.

        Returns the joint density function, or the marginal density 
        function for the first $k$ variables or the last $k$ variables, 
        evaluated at a set of samples.
        
        Parameters
        ----------
        ms:
            An $n \times k$ matrix (where $1 \leq k \leq d$) containing 
            samples from the approximation domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        fms:
            An $n$-dimensional vector containing the value of the 
            approximation to the target density evaluated at each 
            element in `xs`.
        
        """
        neglogfms = self.eval_potential(ms, subset)
        fms = torch.exp(-neglogfms)
        return fms
    
    def _eval_rt(self, xs: Tensor, subset: str) -> Tensor:
        r"""Evaluates the Rosenblatt transport.

        Returns the joint Rosenblatt transport, or the marginal 
        Rosenblatt transport for the first $k$ variables or the last 
        $k$ variables, evaluated at a set of samples.

        Parameters
        ----------
        xs: 
            An $n \times k$ matrix (where $1 \leq k \leq d$) containing 
            samples from the approximation domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        
        Returns
        -------
        zs:
            An $n \times k$ matrix containing the corresponding 
            samples, from the unit hypercube, after applying the 
            Rosenblatt transport.

        """
        direction = SUBSET2DIRECTION[subset]
        indices = self._get_transform_indices(xs.shape[1], direction)
        ls = self.approx.bases.approx2local(xs, indices)[0]
        zs = self._eval_rt_local(ls, direction)
        return zs

    def eval_rt(self, ms: Tensor, subset: str) -> Tensor:
        r"""Evaluates the Rosenblatt transport.

        Returns the joint Rosenblatt transport, or the marginal 
        Rosenblatt transport for the first $k$ variables or the last 
        $k$ variables, evaluated at a set of samples.

        Parameters
        ----------
        xs: 
            An $n \times k$ matrix (where $1 \leq k \leq d$) containing 
            samples from the approximation domain.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        
        Returns
        -------
        zs:
            An $n \times k$ matrix containing the corresponding 
            samples, from the unit hypercube, after applying the 
            Rosenblatt transport.

        """
        xs = self.preconditioner.Q_inv(ms, subset)
        zs = self._eval_rt(xs, subset)
        return zs
    
    def _eval_irt(self, zs: Tensor, subset: str) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the inverse Rosenblatt transport.
        
        Returns the joint inverse Rosenblatt transport, or the marginal 
        inverse Rosenblatt transport for the first $k$ variables or the 
        last $k$ variables, evaluated at a set of samples.
        
        Parameters
        ----------
        zs: 
            An $n \times k$ matrix containing samples from the unit 
            hypercube.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        
        Returns
        -------
        xs: 
            An $n \times k$ matrix containing the corresponding samples 
            from the approximation to the target density function.
        neglogfxs: 
            An $n$-dimensional vector containing the approximation to 
            the potential function evaluated at each sample in `xs`.
        
        """
        direction = SUBSET2DIRECTION[subset]
        indices = self._get_transform_indices(zs.shape[1], direction)
        ls, neglogfls = self._eval_irt_local(zs, direction)
        xs, dxdls = self.bases.local2approx(ls, indices)
        neglogfxs = neglogfls + dxdls.log().sum(dim=1)
        return xs, neglogfxs

    def eval_irt(self, zs: Tensor, subset: str) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the inverse Rosenblatt transport.
        
        Returns the joint inverse Rosenblatt transport, or the marginal 
        inverse Rosenblatt transport for the first $k$ variables or the 
        last $k$ variables, evaluated at a set of samples.
        
        Parameters
        ----------
        zs: 
            An $n \times k$ matrix containing samples from the unit 
            hypercube.
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).
        
        Returns
        -------
        xs: 
            An $n \times k$ matrix containing the corresponding samples 
            from the approximation to the target density function.
        neglogfxs: 
            An $n$-dimensional vector containing the approximation to 
            the potential function evaluated at each sample in `xs`.
        
        """
        xs, neglogfxs = self._eval_irt(zs, subset)

        # Map samples back into actual domain
        ms = self.preconditioner.Q(xs, subset)
        neglogabsdet_ms = self.preconditioner.neglogdet_Q_inv(ms, subset)
        neglogfms = neglogfxs + neglogabsdet_ms

        return ms, neglogfms
    
    def eval_cirt(
        self, 
        xs: Tensor, 
        zs: Tensor, 
        subset: str
    ) -> Tuple[Tensor, Tensor]:
        r"""Evaluates the conditional inverse Rosenblatt transport.

        Returns the conditional inverse Rosenblatt transport evaluated
        at a set of samples in the approximation domain. 
        
        The conditional inverse Rosenblatt transport takes the form
        $$Y|X = R^{-1}(R_{k}(X), Z),$$
        where $X$ is a $k$-dimensional random variable, $Z$ is an 
        $n-k$-dimensional uniform random variable, $R(\,\cdot\,)$ 
        denotes the (full) Rosenblatt transport, and $R_{k}(\,\cdot\,)$ 
        denotes the Rosenblatt transport for the first (or last) $k$ 
        variables.
        
        Parameters
        ----------
        xs:
            An $n \times k$ matrix containing samples from the 
            approximation domain.
        zs:
            An $n \times (d-k)$ matrix containing samples from the unit 
            hypercube of dimension $d-k$.
        subset: 
            Whether `xs` corresponds to the first $k$ variables 
            (`subset='first'`) of the approximation, or the last $k$ 
            variables (`subset='last'`).
        
        Returns
        -------
        ys:
            An $n \times (d-k)$ matrix containing the realisations of 
            $Y$ corresponding to the values of `zs` after applying the 
            conditional inverse Rosenblatt transport.
        neglogfys:
            An $n$-dimensional vector containing the potential function 
            of the approximation to the conditional density of 
            $Y \textbar X$ evaluated at each sample in `ys`.
    
        """
        
        n_zs, d_zs = zs.shape
        n_xs, d_xs = xs.shape

        if d_zs == 0 or d_xs == 0:
            msg = "The dimensions of both X and Z must be at least 1."
            raise Exception(msg)
        
        if d_zs + d_xs != self.dim:
            msg = ("The dimensions of X and Z must sum " 
                   + "to the dimension of the approximation.")
            raise Exception(msg)
        
        if n_zs != n_xs: 
            if n_xs != 1:
                msg = "The number of samples of X and Z must be equal."
                raise Exception(msg)
            xs = xs.repeat(n_zs, 1)
        
        direction = SUBSET2DIRECTION[subset]
        if direction == Direction.FORWARD:
            inds_x = torch.arange(d_xs)
            inds_z = torch.arange(d_xs, self.dim)
        elif direction == Direction.BACKWARD:
            inds_x = torch.arange(d_zs, self.dim)
            inds_z = torch.arange(d_zs)
        
        ls_x = self.bases.approx2local(xs, inds_x)[0]
        ls_y, neglogfys = self._eval_cirt_local(ls_x, zs, direction)
        ys, dydlys = self.bases.local2approx(ls_y, inds_z)
        neglogfys += dydlys.log().sum(dim=1)

        return ys, neglogfys
    
    def eval_potential_grad(
        self, 
        xs: Tensor, 
        method: str = "autodiff",
        subset: str | None = None
    ) -> Tensor:
        r"""Evaluates the gradient of the potential function.
        
        Parameters
        ----------
        xs:
            An $n \times k$ matrix containing samples from the 
            approximation domain.
        method: 
            The method by which to compute the gradient. This can be 
            `autodiff`, or `manual`. Generally, `manual` is faster than 
            `autodiff`, but can only be used to evaluate the gradient 
            of the full potential function (*i.e.*, when $k=d$).
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        grads:
            An $n \times k$ matrix containing the gradient of the 
            potential function evaluated at each sample in `xs`.

        """

        method = method.lower()
        if method not in ("manual", "autodiff"):
            raise Exception("Unknown method.")

        if method == "autodiff":
            TTFunc._check_sample_dim(xs, self.dim)
            grad = self._eval_potential_grad_autodiff(xs, subset)
            return grad
        
        TTFunc._check_sample_dim(xs, self.dim, strict=True)
        ls, dldxs = self.bases.approx2local(xs)
        grad = self._eval_potential_grad_local(ls)
        grad *= dldxs
        return grad

    def eval_rt_jac(
        self, 
        xs: Tensor, 
        method: str = "autodiff",
        subset: str = "first"
    ) -> Tensor:
        r"""Evaluates the Jacobian of the Rosenblatt transport.

        Evaluates the Jacobian of the mapping $Z = R(X)$, where $Z$ is 
        a standard $k$-dimensional uniform random variable and $X$ is 
        the approximation to the target random variable. 

        Note that element $J_{ij}$ of the Jacobian is given by
        $$J_{ij} = \frac{\partial z_{i}}{\partial x_{j}}.$$

        Parameters
        ----------
        xs:
            An $n \times d$ matrix containing a set of samples from the 
            approximation domain.
        method:
            The method by which to compute the Jacobian. This can be 
            `autodiff`, or `manual`. Generally, `manual` is faster than 
            `autodiff`, but can only be used to evaluate the Jacobian 
            of the full Rosenblatt transport (*i.e.*, when $k=d$).
        subset: 
            If the samples contain a subset of the variables, (*i.e.,* 
            $k < d$), whether they correspond to the first $k$ 
            variables (`subset='first'`) or the last $k$ variables 
            (`subset='last'`).

        Returns
        -------
        Jacs:
            A $k \times n \times k$ tensor, where element $ijk$ 
            contains element $ik$ of the Jacobian for the $j$th sample 
            in `xs`.

        """

        direction = SUBSET2DIRECTION[subset]
        method = method.lower()
        if method not in ("manual", "autodiff"):
            raise Exception("Unknown method.")

        if method == "autodiff":
            TTFunc._check_sample_dim(xs, self.dim)
            Jacs = self._eval_rt_jac_autodiff(xs, subset)
            return Jacs
        
        TTFunc._check_sample_dim(xs, self.dim, strict=True)
        ls, dldxs = self.bases.approx2local(xs)
        Jacs = self._eval_rt_jac_local(ls, direction)
        for k in range(self.dim):
            Jacs[:, :, k] *= dldxs[:, k]
        return Jacs

    def random(self, n: int) -> Tensor: 
        """Generates a set of random samples. 
        
        Parameters
        ----------
        n:  
            The number of samples to generate.

        Returns
        -------
        xs:
            The generated samples.
        
        """
        zs = torch.rand(n, self.dim)
        xs = self.eval_irt(zs)[0]
        return xs 
    
    def sobol(self, n: int) -> Tensor:
        """Generates a set of QMC samples.
        
        Parameters
        ----------
        n:
            The number of samples to generate.
        
        Returns
        -------
        xs:
            The generated samples.

        """
        S = SobolEngine(dimension=self.dim)
        zs = S.draw(n)
        xs = self.eval_irt(zs)[0]
        return xs
    

class SIRT(AbstractSIRT):
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

    def __init__(
        self, 
        potential: Callable[[Tensor], Tensor], 
        preconditioner: Preconditioner,
        bases: Basis1D | List[Basis1D] | None = None,
        prev_approx: TTFunc | None = None,
        options: TTOptions | None = None, 
        input_data: InputData | None = None, 
        tt_data: TTData | None = None,
        defensive: float = 1e-8
    ):
        
        def target_func(ls: Tensor) -> Tensor:
            """Returns the square root of the ratio between the target 
            density and the weighting function evaluated at a set of 
            points in the local domain.
            """

            xs = self.bases.local2approx(ls)[0]
            neglogfxs = self.potential(xs)
            neglogwxs = self.bases.eval_measure_potential(xs)[0]
            
            # The ratio of f and w is invariant to changes of coordinate
            gs = torch.exp(-0.5 * (neglogfxs - neglogwxs))
            check_finite(gs)
            return gs
        
        if bases is None and prev_approx is None:
            msg = ("Must pass in a previous approximation or a set of "
                   + "approximation bases.")
            raise Exception(msg)

        if prev_approx is not None:
            bases = prev_approx.bases.polys
            options = prev_approx.options
            tt_data = prev_approx.tt_data

        if options is None:
            options = TTOptions()
        
        if input_data is None:
            input_data = InputData()

        domain = preconditioner.reference.domain
        dim = preconditioner.dim

        self.potential = potential
        self.preconditioner = preconditioner
        self.reference = preconditioner.reference
        self.bases = ApproxBases(bases, domain, dim)
        self.options = options 
        self.input_data = input_data
        self.tt_data = tt_data
        self.defensive = defensive
        self.oned_cdfs = self._construct_cdfs(self.options.cdf_tol)

        self.approx = TTFunc(
            target_func, 
            self.bases,
            options=self.options, 
            input_data=self.input_data,
            tt_data=self.tt_data
        )
        self.approx._cross()
        if self.approx.use_amen:
            self.approx._round()  # why?

        # Compute coefficient tensors and marginalisation coefficents
        self.Bs_f: Dict[int, Tensor] = {}
        self.Rs_f: Dict[int, Tensor] = {}
        self.Bs_b: Dict[int, Tensor] = {}
        self.Rs_b: Dict[int, Tensor] = {}
        self._marginalise_forward()
        self._marginalise_backward()
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
        return


class SavedSIRT(AbstractSIRT):

    def __init__(
        self, 
        data: Dict,
        preconditioner: Preconditioner, 
        bases: List[Basis1D],
        options: TTOptions
    ):
        
        domain = preconditioner.reference.domain
        dim = preconditioner.dim

        direction = data["direction"]  # TODO: actually get the enum
        cores = {int(k): v for k, v in data["cores"].items()}

        self.preconditioner = preconditioner
        self.reference = preconditioner.reference
        self.bases = ApproxBases(bases, domain, dim)
        self.options = options
        self.input_data = InputData(data["xs_samp"], data["xs_debug"], data["fxs_debug"])
        self.tt_data = TTData(direction, cores)
        self.Bs_f = {int(k): v for k, v in data["Bs_f"].items()}
        self.Rs_f = {int(k): v for k, v in data["Rs_f"].items()}
        self.Bs_b = {int(k): v for k, v in data["Bs_b"].items()}
        self.Rs_b = {int(k): v for k, v in data["Rs_b"].items()}
        self.defensive = data["defensive"]
        self.z_func = self.Rs_f[0].square().sum()
        self.oned_cdfs = self._construct_cdfs(self.options.cdf_tol)

        self.approx = SavedTTFunc(
            self.bases,
            options=self.options,
            input_data=self.input_data,
            tt_data=self.tt_data
        )

        return