import torch 
from torch import Tensor 

from ..density_estimators import EmpiricalMarginals
from ..preconditioners import Preconditioner
from ..references import GaussianReference, Reference


class EmpiricalCDFMapping(Preconditioner):

    def __init__(
        self, 
        cdfs: EmpiricalMarginals,
        reference: Reference | None = None
    ):

        if reference is None:
            reference = GaussianReference()

        def Q(xs: Tensor) -> Tensor:
            zs = self.reference.eval_cdf(xs)[0]
            ms = cdfs.invert_cdf(zs)
            return ms
        
        def Q_inv(ms: Tensor) -> Tensor:
            zs = cdfs.eval_cdf(ms)[0]
            xs = self.reference.invert_cdf(zs)
            return xs

        def neglogdet_Q(xs: Tensor) -> Tensor:
            ms = self.Q(xs)
            return reference.eval_potential(xs)[0] - cdfs.eval_potential(ms)

        def neglogdet_Q_inv(ms: Tensor) -> Tensor:
            xs = self.Q_inv(ms)
            return cdfs.eval_potential(ms) - reference.eval_potential(xs)[0]
        
        Preconditioner.__init__(
            self,
            reference=reference,
            Q=Q,
            Q_inv=Q_inv,
            neglogdet_Q=neglogdet_Q,
            neglogdet_Q_inv=neglogdet_Q_inv, 
            dim=cdfs.dim
        )

        return