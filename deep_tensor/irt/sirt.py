from typing import Callable, Dict, Tuple

import torch
from torch import Tensor

from ..ftt import Direction, FTT
from ..linalg import n_mode_prod, unfold_left, unfold_right
from ..polynomials import construct_cdf
from ..references import Reference
from ..tools import estimate_dhell


SUBSET2DIRECTION = {
    "first": Direction.FORWARD,
    "last": Direction.BACKWARD
}


class SIRT():
    """Squared inverse Rosenblatt transport.
    
    Parameters
    ----------
    eval_neglogtarget:
        A function that receives an n * d matrix of samples and 
        returns an n-dimensional vector containing the potential 
        function of the target density evaluated at each sample.
    ftt:
        The functional tensor train to use to approximate the 
        square root of the ratio between the target density and 
        weighting function.
    reference:
        The reference density.
    domain: 
        The domain of the reference.
    defensive:
        The defensive parameter.
    cdf_tol:
        The tolerance used when solving the rootfinding problem to 
        evaluate the inverse of each conditional CDF.
    num_error_samples:
        The number of samples to use to compute and importance sampling 
        estimate of the Hellinger distance between the target function 
        and the SIRT approximation.
    device:
        The device to carry out computations on.

    """

    def __init__(
        self, 
        eval_neglogtarget: Callable[[Tensor], Tensor], 
        ftt: FTT,
        dim: int,
        reference: Reference,
        defensive: float,
        cdf_tol: float,
        num_error_samples: int,
        device: torch.device = torch.get_default_device()
    ):

        self.eval_neglogtarget = eval_neglogtarget
        self.ftt = ftt
        self.basis = self.ftt.basis
        self.dim = dim
        self.domain = reference.domain
        self.defensive = defensive
        self.cdf = construct_cdf(self.basis, error_tol=cdf_tol) 
        self.ftt.approximate(self._target_func, dim, reference)
        self.device = device

        # Precompute coefficient tensors and marginalisation 
        # coefficents, from the first core to the last and the last 
        # core to the first.
        self._Bs_f: Dict[int, Tensor] = {}
        self._Rs_f: Dict[int, Tensor] = {}
        self._Bs_b: Dict[int, Tensor] = {}
        self._Rs_b: Dict[int, Tensor] = {}
        self._marginalise_forward()
        self._marginalise_backward()
        
        self.dhell = self._estimate_dhell(num_error_samples)
        return
    
    @property
    def z(self) -> Tensor:
        return (1.0 + self.defensive) * self.z_func

    @property 
    def coef_defensive(self) -> Tensor:
        # Note: this is a slight change from the defensive parameter 
        # defined in @CuiDolgov2022. The defensive parameter now scales 
        # according to the normalising constant of the FTT approximation 
        # to the target density.
        return self.defensive * self.z_func
    
    @property 
    def num_eval(self) -> int:
        return self.ftt.num_eval
    
    @property 
    def num_eval_construction(self) -> int:
        return self.ftt.num_eval_construction
    
    def _eval_measure_potential(self, xs: Tensor) -> Tensor:
        """Computes the target potential function for a set of samples 
        from the approximation domain.        
        """
        ls, dldxs = self.domain.approx2local(xs)
        neglogwls = -self.basis.eval_log_measure(ls).sum(dim=1)
        neglogwxs = neglogwls - dldxs.log().sum(dim=1)        
        return neglogwxs

    def _target_func(self, ls: Tensor) -> Tensor:
        """Returns the square root of the ratio between the target 
        density and the weighting function evaluated at a set of points 
        in the local domain (note: this ratio is invariant to changes 
        of coordinate).
        """
        xs = self.domain.local2approx(ls)[0]
        neglogfxs = self.eval_neglogtarget(xs)
        neglogwxs = self._eval_measure_potential(xs)
        gs = torch.exp(-0.5 * (neglogfxs - neglogwxs))
        return gs
    
    def _marginalise_forward(self) -> None:
        """Computes each coefficient tensor required to evaluate the 
        marginal functions in each dimension, by iterating over the 
        dimensions of the approximation from last to first.
        """

        self._Rs_f[self.dim] = torch.tensor([[1.0]], device=self.device)
        cores = self.ftt.cores

        for k in range(self.dim-1, -1, -1):
            self._Bs_f[k] = n_mode_prod(cores[k], self._Rs_f[k+1].T, n=2)
            C_k = n_mode_prod(self._Bs_f[k], self.basis.mass_R.T, n=1)
            C_k = unfold_right(C_k)
            self._Rs_f[k] = torch.linalg.qr(C_k, mode="reduced")[1].T

        self.z_func = self._Rs_f[0].square().sum()
        return 
    
    def _marginalise_backward(self) -> None:
        """Computes each coefficient tensor required to evaluate the 
        marginal functions in each dimension, by iterating over the 
        dimensions of the approximation from first to last.
        """
        
        self._Rs_b[-1] = torch.tensor([[1.0]], device=self.device)
        cores = self.ftt.cores

        for k in range(self.dim):
            self._Bs_b[k] = n_mode_prod(cores[k], self._Rs_b[k-1], n=0)
            C_k = n_mode_prod(self._Bs_b[k], self.basis.mass_R, n=1)
            C_k = unfold_left(C_k)
            self._Rs_b[k] = torch.linalg.qr(C_k, mode="reduced")[1]

        self.z_func = self._Rs_b[self.dim-1].square().sum()
        return

    def _estimate_dhell(self, num_samples: int) -> float | None:
        """Computes an estimate of the Hellinger distance between 
        the ratio function and SIRT approximation.
        """
        if num_samples == 0:
            return None
        zs = torch.rand(num_samples, self.dim)
        us, neglogfus = self.eval_irt(zs, subset="first")
        neglogfus_exact = self.eval_neglogtarget(us)
        dhell = estimate_dhell(neglogfus, neglogfus_exact)
        return float(dhell)

    def _eval_rt_local_forward(self, ls: Tensor) -> Tensor:

        num_ls, dim_ls = ls.shape
        zs = torch.zeros_like(ls)
        Gs_prod = torch.ones((num_ls, 1), device=ls.device)

        cores = self.ftt.cores
        Bs = self._Bs_f 
            
        for k in range(dim_ls):
            
            # Compute (unnormalised) conditional PDF for each sample
            Ps = FTT.eval_core(self.basis, Bs[k], self.cdf.nodes)
            gs = torch.einsum("jl, ilk -> ijk", Gs_prod, Ps)
            ps = gs.square().sum(dim=2) + self.coef_defensive

            # Evaluate CDF to obtain corresponding uniform variates
            zs[:, k] = self.cdf.eval_cdf(ps, ls[:, k])

            # Compute incremental product of tensor cores for each sample
            Gs = FTT.eval_core(self.basis, cores[k], ls[:, k])
            Gs_prod = torch.einsum("il, ilk -> ik", Gs_prod, Gs)

        return zs
    
    def _eval_rt_local_backward(self, ls: Tensor) -> Tensor:

        num_ls, dim_ls = ls.shape
        zs = torch.zeros_like(ls)
        d_min = self.dim - dim_ls
        Gs_prod = torch.ones((1, num_ls), device=ls.device)

        cores = self.ftt.cores
        Bs = self._Bs_b 

        for i, k in enumerate(range(self.dim-1, d_min-1, -1), start=1):

            # Compute (unnormalised) conditional PDF for each sample
            Ps = FTT.eval_core(self.basis, Bs[k], self.cdf.nodes)
            gs = torch.einsum("ijl, lk -> ijk", Ps, Gs_prod)
            ps = gs.square().sum(dim=1) + self.coef_defensive

            # Evaluate CDF to obtain corresponding uniform variates
            zs[:, -i] = self.cdf.eval_cdf(ps, ls[:, -i])
            
            # Compute incremental product of tensor cores for each sample
            Gs = FTT.eval_core(self.basis, cores[k], ls[:, -i])
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
        gs = torch.ones((n_zs, 1), device=zs.device)

        Bs = self._Bs_f

        for k in range(d_zs):
            
            Ps = FTT.eval_core(self.basis, Bs[k], self.cdf.nodes)
            gls = n_mode_prod(Ps, gs, n=1)
            ps = gls.square().sum(dim=2) + self.coef_defensive
            ls[:, k] = self.cdf.invert_cdf(ps, zs[:, k])

            Gs = FTT.eval_core(self.basis, self.ftt.cores[k], ls[:, k])
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
        gs = torch.ones((n_zs, 1), device=zs.device)
        d_min = self.dim - d_zs

        cores = self.ftt.cores
        Bs = self._Bs_b

        for i, k in enumerate(range(self.dim-1, d_min-1, -1), start=1):

            Ps = FTT.eval_core_rev(self.basis, Bs[k], self.cdf.nodes)
            gls = n_mode_prod(Ps, gs, n=1)
            ps = gls.square().sum(dim=2) + self.coef_defensive
            ls[:, -i] = self.cdf.invert_cdf(ps, zs[:, -i])

            Gs = FTT.eval_core_rev(self.basis, cores[k], ls[:, -i])
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
        neglogpls = -(gs_sq + self.coef_defensive).log()
        neglogwls = -self.basis.eval_log_measure(ls).sum(dim=1)
        neglogfls = self.z.log() + neglogpls + neglogwls
        return ls, neglogfls

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
            gs = self.ftt(ls, direction=direction)
            gs_sq = (gs @ self._Rs_f[dim_l]).square().sum(dim=1)
        else:
            gs = self.ftt(ls, direction=direction)
            gs_sq = (self._Rs_b[self.dim-dim_l-1] @ gs.T).square().sum(dim=0)
        
        neglogwls = -self.basis.eval_log_measure(ls).sum(dim=1)
        neglogfls = self.z.log() - (gs_sq + self.coef_defensive).log() + neglogwls
        return neglogfls
    
    def eval_potential(self, xs: Tensor, subset: str) -> Tensor:
        """Returns the joint potential function, or the marginal 
        potential function for the first k variables or the last k 
        variables, evaluated at a set of samples.

        Parameters
        ----------
        xs:
            An n * k matrix (where 1 < k < d) containing samples from 
            the approximation domain.
        subset: 
            If the samples contain a subset of the variables, (i.e.,  
            k < d), whether they correspond to the first k variables 
            (subset='first') or the last k variables (subset='last').
        
        Returns
        -------
        neglogfxs:
            The potential function of the approximation to the target 
            density evaluated at each sample in xs.

        """
        direction = SUBSET2DIRECTION[subset]
        ls, dldxs = self.domain.approx2local(xs)
        neglogfls = self._eval_potential_local(ls, direction)
        neglogfxs = neglogfls - dldxs.log().sum(dim=1)
        return neglogfxs

    def eval_rt(self, xs: Tensor, subset: str) -> Tensor:
        """Returns the joint Rosenblatt transport, or the marginal 
        Rosenblatt transport for the first k variables or the last k 
        variables, evaluated at a set of samples.

        Parameters
        ----------
        xs: 
            An n * k matrix (where 1 < k < d) containing samples from 
            the approximation domain.
        subset: 
            If the samples contain a subset of the variables, (i.e., 
            k < d), whether they correspond to the first k variables 
            (`subset='first'`) or the last k variables 
            (`subset='last'`).
        
        Returns
        -------
        zs:
            An n * k matrix containing the corresponding samples, from 
            the unit hypercube, after applying the Rosenblatt transport.

        """
        direction = SUBSET2DIRECTION[subset]
        ls = self.domain.approx2local(xs)[0]
        zs = self._eval_rt_local(ls, direction)
        return zs

    def eval_irt(self, zs: Tensor, subset: str) -> Tuple[Tensor, Tensor]:
        """Returns the joint inverse Rosenblatt transport, or the 
        marginal inverse Rosenblatt transport for the first k variables 
        or the last k variables, evaluated at a set of samples.
        
        Parameters
        ----------
        zs: 
            An n * k matrix containing samples from the unit hypercube.
        subset: 
            If the samples contain a subset of the variables, (i.e., 
            k < d), whether they correspond to the first k variables 
            (subset='first'`) or the last k variables (subset='last').
        
        Returns
        -------
        xs: 
            An n * k matrix containing the corresponding samples from 
            the approximation to the target density function.
        neglogfxs: 
            An n-dimensional vector containing the approximation to the 
            potential function evaluated at each sample in xs.
        
        """
        direction = SUBSET2DIRECTION[subset]
        ls, neglogfls = self._eval_irt_local(zs, direction)
        xs, dxdls = self.domain.local2approx(ls)
        neglogfxs = neglogfls + dxdls.log().sum(dim=1)
        return xs, neglogfxs