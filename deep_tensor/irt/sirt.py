from typing import Callable, Dict, Tuple

import torch
from torch import Tensor

from ..domains import Domain
from ..ftt import ApproxBases, Direction, FTT
from ..linalg import batch_mul, n_mode_prod, unfold_left, unfold_right
from ..polynomials import CDF1D, construct_cdf


SUBSET2DIRECTION = {
    "first": Direction.FORWARD,
    "last": Direction.BACKWARD
}


class SIRT():
    r"""Squared inverse Rosenblatt transport.
    
    Parameters
    ----------
    potential:
        A function that receives an $n \times d$ matrix of samples and 
        returns an $n$-dimensional vector containing the potential 
        function of the target density evaluated at each sample.
    ftt:
        TODO
    defensive: 
        TODO
    defensive:
        The defensive parameter, $\tau$, which ensures that the tails
        of the approximation are sufficiently heavy.
    cdf_tol:
        TODO

    References
    ----------
    Cui, T and Dolgov, S (2022). *[Deep composition of tensor-trains 
    using squared inverse Rosenblatt transports](https://doi.org/10.1007/s10208-021-09537-5).* 
    Foundations of Computational Mathematics, **22**, 1863--1922.

    """

    def __init__(
        self, 
        target_func: Callable[[Tensor], Tensor], 
        ftt: FTT,
        reference,
        domain: Domain,
        defensive: float,
        cdf_tol: float
    ):

        self.potential = target_func
        self.ftt = ftt
        self.bases = self.ftt.bases
        self.dim = self.ftt.dim
        self.domain = domain
        self.defensive = defensive
        self.cdfs = self.construct_cdfs(self.bases, cdf_tol)

        self.ftt.approximate(self._target_func, reference)

        # Compute coefficient tensors and marginalisation coefficents, 
        # from the first core to the last and the last core to the first
        self._Bs_f: Dict[int, Tensor] = {}
        self._Rs_f: Dict[int, Tensor] = {}
        self._Bs_b: Dict[int, Tensor] = {}
        self._Rs_b: Dict[int, Tensor] = {}
        self._marginalise_forward()
        self._marginalise_backward()
        return
    
    @property
    def z(self) -> Tensor:
        return self.defensive + self.z_func
    
    @property 
    def num_eval(self) -> int:
        return self.ftt.num_eval

    @staticmethod
    def construct_cdfs(bases: ApproxBases, tol: float) -> Dict[int, CDF1D]:
        cdfs = {}
        for k in range(bases.dim):
            cdfs[k] = construct_cdf(bases[k], error_tol=tol) 
        return cdfs
    
    def local2approx(
        self, 
        ls: Tensor, 
        inds: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Maps a set of samples drawn distributed in (a subset of) the 
        local domain to the approximation domain.
        """
        if inds is None:
            inds = torch.arange(self.dim)
        ApproxBases._check_indices_shape(inds, ls)
        xs = torch.empty_like(ls)
        dxdls = torch.empty_like(ls)
        for i, ls_i in enumerate(ls.T):
            xs[:, i], dxdls[:, i] = self.domain.local2approx(ls_i)
        return xs, dxdls
    
    def approx2local(
        self, 
        xs: Tensor, 
        inds: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Maps a set of samples from (a subset of) the approximation 
        domain to the local domain.
        """
        if inds is None:
            inds = torch.arange(self.dim)
        ApproxBases._check_indices_shape(inds, xs)
        ls = torch.empty_like(xs)
        dldxs = torch.empty_like(xs)
        for i, xs_i in enumerate(xs.T):
            ls[:, i], dldxs[:, i] = self.domain.approx2local(xs_i)
        return ls, dldxs
    
    def eval_measure_potential(
        self, 
        xs: Tensor, 
        inds: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Computes the target potential function and its gradient for 
        a set of samples from the approximation domain.
        
        Parameters
        ----------
        xs:
            An n * d matrix containing a set of samples from the 
            approximation domain.
        inds:
            The indices corresponding to the dimensions of the 
            approximation domain the samples live in (can be a subset 
            of {1, 2, ..., d}). 
        
        Returns
        -------
        neglogwxs:
            An n-dimensional vector containing the weighting function 
            evaluated at each element of xs.
        negloggradwxs:
            An n * d matrix containing the gradient of the negative 
            logarithm of each weighting function evaluated at each 
            element of xs.
        
        """
        
        if inds is None:
            inds = torch.arange(self.dim)
        ApproxBases._check_indices_shape(inds, xs)

        ls, dldxs = self.approx2local(xs, inds)
        
        neglogwls = self.bases.eval_measure_potential(ls, inds)
        neglogwxs = neglogwls - dldxs.log().sum(dim=1)
        
        gradneglogwls = self.bases.eval_measure_potential_grad(ls, inds)
        gradneglogwxs = gradneglogwls * dldxs
        return neglogwxs, gradneglogwxs

    def _target_func(self, ls: Tensor) -> Tensor:
        """Returns the square root of the ratio between the target 
        density and the weighting function evaluated at a set of points 
        in the local domain (note that this is invariant to changes of 
        coordinate).
        """
        xs = self.local2approx(ls)[0]
        neglogfxs = self.potential(xs)
        neglogwxs = self.eval_measure_potential(xs)[0]
        gs = torch.exp(-0.5 * (neglogfxs - neglogwxs))
        return gs

    def _marginalise_forward(self) -> None:
        """Computes each coefficient tensor required to evaluate the 
        marginal functions in each dimension, by iterating over the 
        dimensions of the approximation from last to first.
        """

        self._Rs_f[self.dim] = torch.tensor([[1.0]])
        cores = self.ftt.cores

        for k in range(self.dim-1, -1, -1):
            self._Bs_f[k] = n_mode_prod(cores[k], self._Rs_f[k+1].T, n=2)
            C_k = n_mode_prod(self._Bs_f[k], self.bases[k].mass_R.T, n=1)
            C_k = unfold_right(C_k)
            self._Rs_f[k] = torch.linalg.qr(C_k, mode="reduced")[1].T

        self.z_func = self._Rs_f[0].square().sum()
        return 
    
    def _marginalise_backward(self) -> None:
        """Computes each coefficient tensor required to evaluate the 
        marginal functions in each dimension, by iterating over the 
        dimensions of the approximation from first to last.
        """
        
        self._Rs_b[-1] = torch.tensor([[1.0]])
        cores = self.ftt.cores

        for k in range(self.dim):
            self._Bs_b[k] = n_mode_prod(cores[k], self._Rs_b[k-1], n=0)
            C_k = n_mode_prod(self._Bs_b[k], self.bases[k].mass_R, n=1)
            C_k = unfold_left(C_k)
            self._Rs_b[k] = torch.linalg.qr(C_k, mode="reduced")[1]

        self.z_func = self._Rs_b[self.dim-1].square().sum()
        return

    def _eval_rt_local_forward(self, ls: Tensor) -> Tensor:

        n_ls, d_ls = ls.shape
        zs = torch.zeros_like(ls)
        Gs_prod = torch.ones((n_ls, 1))

        cores = self.ftt.cores
        Bs = self._Bs_f 
            
        for k in range(d_ls):
            
            # Compute (unnormalised) conditional PDF for each sample
            Ps = FTT.eval_core(self.bases[k], Bs[k], self.cdfs[k].nodes)
            gs = torch.einsum("jl, ilk -> ijk", Gs_prod, Ps)
            ps = gs.square().sum(dim=2) + self.defensive

            # Evaluate CDF to obtain corresponding uniform variates
            zs[:, k] = self.cdfs[k].eval_cdf(ps, ls[:, k])

            # Compute incremental product of tensor cores for each sample
            Gs = FTT.eval_core(self.bases[k], cores[k], ls[:, k])
            Gs_prod = torch.einsum("il, ilk -> ik", Gs_prod, Gs)

        return zs
    
    def _eval_rt_local_backward(self, ls: Tensor) -> Tensor:

        n_ls, d_ls = ls.shape
        zs = torch.zeros_like(ls)
        d_min = self.dim - d_ls
        Gs_prod = torch.ones((1, n_ls))

        cores = self.ftt.cores
        Bs = self._Bs_b 

        for i, k in enumerate(range(self.dim-1, d_min-1, -1), start=1):

            # Compute (unnormalised) conditional PDF for each sample
            Ps = FTT.eval_core(self.bases[k], Bs[k], self.cdfs[k].nodes)
            gs = torch.einsum("ijl, lk -> ijk", Ps, Gs_prod)
            ps = gs.square().sum(dim=1) + self.defensive

            # Evaluate CDF to obtain corresponding uniform variates
            zs[:, -i] = self.cdfs[k].eval_cdf(ps, ls[:, -i])
            
            # Compute incremental product of tensor cores for each sample
            Gs = FTT.eval_core(self.bases[k], cores[k], ls[:, -i])
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

        Bs = self._Bs_f

        for k in range(d_zs):
            
            Ps = FTT.eval_core(self.bases[k], Bs[k], self.cdfs[k].nodes)
            gls = n_mode_prod(Ps, gs, n=1)
            ps = gls.square().sum(dim=2) + self.defensive
            ls[:, k] = self.cdfs[k].invert_cdf(ps, zs[:, k])

            Gs = FTT.eval_core(self.bases[k], self.ftt.cores[k], ls[:, k])
            gs = torch.einsum("il, ilk -> ik", gs, Gs)
        
        gs_sq = (gs @ self._Rs_f[d_zs]).square().sum(dim=1)
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

        cores = self.ftt.cores
        Bs = self._Bs_b

        for i, k in enumerate(range(self.dim-1, d_min-1, -1), start=1):

            Ps = FTT._eval_core_231(self.bases[k], Bs[k], self.cdfs[k].nodes)
            gls = n_mode_prod(Ps, gs, n=1)
            ps = gls.square().sum(dim=2) + self.defensive
            ls[:, -i] = self.cdfs[k].invert_cdf(ps, zs[:, -i])

            Gs = FTT._eval_core_231(self.bases[k], cores[k], ls[:, -i])
            gs = torch.einsum("il, ilk -> ik", gs, Gs)

        gs_sq = (self._Rs_b[d_min-1] @ gs.T).square().sum(dim=0)
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
        neglogwls = self.bases.eval_measure_potential(ls, indices)
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

        cores = self.ftt.cores
        Bs = self._Bs_f
        
        Gs_prod = torch.ones((n_xs, 1, 1))

        for k in range(d_xs-1):
            Gs = FTT._eval_core_213(self.bases[k], cores[k], ls_x[:, k])
            Gs_prod = batch_mul(Gs_prod, Gs)
        
        k = d_xs-1

        Ps = FTT._eval_core_213(self.bases[k], Bs[k], ls_x[:, k])
        gs_marg = batch_mul(Gs_prod, Ps)
        ps_marg = gs_marg.square().sum(dim=(1, 2)) + self.defensive

        Gs = FTT._eval_core_213(self.bases[k], cores[k], ls_x[:, k])
        Gs_prod = batch_mul(Gs_prod, Gs)

        # Generate conditional samples
        for i, k in enumerate(range(d_xs, self.dim)):
            
            Ps = FTT._eval_core_213(self.bases[k], Bs[k], self.cdfs[k].nodes)
            gs = torch.einsum("mij, ljk -> lmk", Gs_prod, Ps)
            ps = gs.square().sum(dim=2) + self.defensive
            ls_y[:, i] = self.cdfs[k].invert_cdf(ps, zs[:, i])

            Gs = FTT._eval_core_213(self.bases[k], cores[k], ls_y[:, i])
            Gs_prod = batch_mul(Gs_prod, Gs)

        ps = Gs_prod.flatten().square() + self.defensive

        indices = d_xs + torch.arange(d_zs)
        neglogwls_y = self.bases.eval_measure_potential(ls_y, indices)
        neglogfls_y = ps_marg.log() - ps.log() + neglogwls_y

        return ls_y, neglogfls_y
    
    def _eval_cirt_local_backward(
        self, 
        ls_x: Tensor, 
        zs: Tensor
    ) -> Tuple[Tensor, Tensor]:

        n_zs, d_zs = zs.shape
        ls_y = torch.zeros_like(zs)

        cores = self.ftt.cores
        Bs = self._Bs_b

        Gs_prod = torch.ones((n_zs, 1, 1))

        for i, k in enumerate(range(self.dim-1, d_zs, -1), start=1):
            Gs = FTT._eval_core_213(self.bases[k], cores[k], ls_x[:, -i])
            Gs_prod = batch_mul(Gs, Gs_prod)

        Ps = FTT._eval_core_213(self.bases[d_zs], Bs[d_zs], ls_x[:, 0])
        gs_marg = batch_mul(Ps, Gs_prod)
        ps_marg = gs_marg.square().sum(dim=(1, 2)) + self.defensive

        Gs = FTT._eval_core_213(self.bases[d_zs], cores[d_zs], ls_x[:, 0])
        Gs_prod = batch_mul(Gs, Gs_prod)

        # Generate conditional samples
        for k in range(d_zs-1, -1, -1):

            Ps = FTT._eval_core_213(self.bases[k], Bs[k], self.cdfs[k].nodes)
            gs = torch.einsum("lij, mjk -> lmi", Ps, Gs_prod)
            ps = gs.square().sum(dim=2) + self.defensive
            ls_y[:, k] = self.cdfs[k].invert_cdf(ps, zs[:, k])

            Gs = FTT._eval_core_213(self.bases[k], cores[k], ls_y[:, k])
            Gs_prod = batch_mul(Gs, Gs_prod)

        ps = Gs_prod.flatten().square() + self.defensive

        indices = torch.arange(d_zs-1, -1, -1)
        neglogwls_y = self.bases.eval_measure_potential(ls_y, indices)
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

        cores = self.ftt.cores

        zs = self._eval_rt_local_forward(ls)
        ls, gs_sq = self._eval_irt_local_forward(zs)
        n_ls = ls.shape[0]
        ps = gs_sq + self.defensive
        neglogws = self.bases.eval_measure_potential(ls)
        ws = torch.exp(-neglogws)
        fs = ps * ws  # Don't need to normalise as derivative ends up being a ratio
        
        Gs_prod = torch.ones((n_ls, 1, 1))
        
        dwdls = {k: torch.ones((n_ls, )) for k in range(self.dim)}
        dGdls = {k: torch.ones((n_ls, 1, 1)) for k in range(self.dim)}
        
        for k in range(self.dim):

            ws_k = self.bases[k].eval_measure(ls[:, k])
            dwdls_k = self.bases[k].eval_measure_deriv(ls[:, k])

            Gs_k = FTT._eval_core_213(self.bases[k], cores[k], ls[:, k])
            dGdls_k = FTT._eval_core_213_deriv(self.bases[k], cores[k], ls[:, k])
            Gs_prod = batch_mul(Gs_prod, Gs_k)
            
            for j in range(self.dim):
                if k == j:
                    dwdls[j] *= dwdls_k
                    dGdls[j] = batch_mul(dGdls[j], dGdls_k)
                else:
                    dwdls[j] *= ws_k
                    dGdls[j] = batch_mul(dGdls[j], Gs_k)
        
        dfdls = torch.zeros_like(ls)
        deriv = torch.zeros_like(ls)
        gs = Gs_prod.sum(dim=(1, 2)) 

        for k in range(self.dim):
            dGdls_k = dGdls[k].sum(dim=(1, 2))
            dfdls[:, k] = ps * dwdls[k] + 2.0 * gs * dGdls_k * ws
            deriv[:, k] = -dfdls[:, k] / fs

        return deriv

    def _eval_rt_jac_local_forward(self, ls: Tensor) -> Tensor:

        cores = self.ftt.cores
        Bs = self._Bs_f

        Gs: Dict[int, Tensor] = {}
        Gs_deriv: Dict[int, Tensor] = {}
        Ps: Dict[int, Tensor] = {}
        Ps_deriv: Dict[int, Tensor] = {}
        Ps_grid: Dict[int, Tensor] = {}

        ps_marg: Dict[int, Tensor] = {}
        ps_marg[-1] = self.z
        ps_marg_deriv: Dict[int, Dict[int, Tensor]] = {}
        ps_grid: Dict[int, Tensor] = {}
        ps_grid_deriv: Dict[int, Dict[int, Tensor]] = {}
        wls: Dict[int, Tensor] = {}

        n_ls = ls.shape[0]
        Jacs = torch.zeros((self.dim, n_ls, self.dim))

        Gs_prod = {} 
        Gs_prod[-1] = torch.ones((n_ls, 1, 1))

        for k in range(self.dim):

            # Evaluate weighting function
            wls[k] = self.bases[k].eval_measure(ls[:, k])

            # Evaluate kth tensor core and derivative
            Gs[k] = FTT._eval_core_213(self.bases[k], cores[k], ls[:, k])
            Gs_deriv[k] = FTT._eval_core_213_deriv(self.bases[k], cores[k], ls[:, k])
            Gs_prod[k] = batch_mul(Gs_prod[k-1], Gs[k])

            # Evaluate kth marginalisation core and derivative
            Ps[k] = FTT._eval_core_213(self.bases[k], Bs[k], ls[:, k])
            Ps_deriv[k] = FTT._eval_core_213_deriv(self.bases[k], Bs[k], ls[:, k])
            Ps_grid[k] = FTT._eval_core_213(self.bases[k], Bs[k], self.cdfs[k].nodes)

            # Evaluate marginal probability for the first k elements of 
            # each sample
            gs = batch_mul(Gs_prod[k-1], Ps[k])
            ps_marg[k] = gs.square().sum(dim=(1, 2)) + self.defensive

            # Compute (unnormalised) marginal PDF at CDF nodes for each sample
            gs_grid = torch.einsum("mij, ljk -> lmik", Gs_prod[k-1], Ps_grid[k])
            ps_grid[k] = gs_grid.square().sum(dim=(2, 3)) + self.defensive

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

        # Populate off-diagonal elements
        for k in range(1, self.dim):
            for j in range(k):
                grad_cond = (ps_grid_deriv[k][j] * ps_marg[k-1] 
                             - ps_grid[k] * ps_marg_deriv[k-1][j]) / ps_marg[k-1].square()
                if self.bases[k].constant_weight:
                    grad_cond *= wls[k]
                Jacs[k, :, j] = self.cdfs[k].eval_int_deriv(grad_cond, ls[:, k])

        return Jacs
    
    def _eval_rt_jac_local_backward(self, ls: Tensor) -> Tensor:

        cores = self.ftt.cores
        Bs = self._Bs_b

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
            wls[k] = self.bases[k].eval_measure(ls[:, k])

            # Evaluate kth tensor core and derivative
            Gs[k] = FTT._eval_core_231(self.bases[k], cores[k], ls[:, k])
            Gs_deriv[k] = FTT._eval_core_231_deriv(self.bases[k], cores[k], ls[:, k])
            Gs_prod[k] = batch_mul(Gs_prod[k+1], Gs[k])

            # Evaluate kth marginalisation core and derivative
            Ps[k] = FTT._eval_core_231(self.bases[k], Bs[k], ls[:, k])
            Ps_deriv[k] = FTT._eval_core_231_deriv(self.bases[k], Bs[k], ls[:, k])
            Ps_grid[k] = FTT._eval_core_231(self.bases[k], Bs[k], self.cdfs[k].nodes)

            # Evaluate marginal probability for the first k elements of 
            # each sample
            gs = batch_mul(Gs_prod[k+1], Ps[k])
            ps_marg[k] = gs.square().sum(dim=(1, 2)) + self.defensive

            # Compute (unnormalised) marginal PDF at CDF nodes for each sample
            gs_grid = torch.einsum("mij, ljk -> lmik", Gs_prod[k+1], Ps_grid[k])
            ps_grid[k] = gs_grid.square().sum(dim=(2, 3)) + self.defensive

        # Derivatives of marginal PDF
        for k in range(1, self.dim):
            ps_marg_deriv[k] = {}

            for j in range(k, self.dim):

                prod = batch_mul(Gs_prod[k+1], Ps[k])
                prod_deriv = torch.ones((n_ls, 1, 1))

                for k_i in range(self.dim-1, k, -1):
                    core = Gs_deriv[k_i] if k_i == j else Gs[k_i]
                    prod_deriv = batch_mul(prod_deriv, core)
                core = Ps_deriv[k] if k == j else Ps[k] 
                prod_deriv = batch_mul(prod_deriv, core)

                ps_marg_deriv[k][j] = 2 * (prod * prod_deriv).sum(dim=(1, 2))

        for k in range(self.dim-1):
            ps_grid_deriv[k] = {}

            for j in range(k+1, self.dim):

                prod = torch.einsum("mij, ljk -> lmik", Gs_prod[k+1], Ps_grid[k])
                prod_deriv = torch.ones((n_ls, 1, 1))

                for k_i in range(self.dim-1, k, -1):
                    core = Gs_deriv[k_i] if k_i == j else Gs[k_i]
                    prod_deriv = batch_mul(prod_deriv, core)
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
                if self.bases[k].constant_weight:
                    grad_cond *= wls[k]
                Jacs[k, :, j] = self.cdfs[k].eval_int_deriv(grad_cond, ls[:, k])
            
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
        raise Exception("Unknown direction encountered.")
    
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
            gs = self.ftt(ls, direction=direction)
            gs_sq = (gs @ self._Rs_f[dim_l]).square().sum(dim=1)
            
        else:
            indices = torch.arange(self.dim-dim_l, self.dim)
            gs = self.ftt(ls, direction=direction)
            gs_sq = (self._Rs_b[self.dim-dim_l-1] @ gs.T).square().sum(dim=0)
        
        neglogwls = self.bases.eval_measure_potential(ls, indices)
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
        ls, dldxs = self.approx2local(xs, indices)
        neglogfls = self._eval_potential_local(ls, direction)
        neglogfxs = neglogfls - dldxs.log().sum(dim=1)
        return neglogfxs

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
        ls = self.approx2local(xs, indices)[0]
        zs = self._eval_rt_local(ls, direction)
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
        xs, dxdls = self.local2approx(ls, indices)
        neglogfxs = neglogfls + dxdls.log().sum(dim=1)
        return xs, neglogfxs