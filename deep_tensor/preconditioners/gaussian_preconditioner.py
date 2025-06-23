from torch import Tensor
from torch import linalg

from .preconditioner import Preconditioner
from ..references import GaussianReference


class GaussianPreconditioner(Preconditioner):
    r"""A mapping between two Gaussian densities.
    
    This preconditioner provides a mapping between the standard 
    Gaussian density and a 

    """

    def __init__(
        self,
        mean: Tensor,
        cov: Tensor, 
        reference: GaussianReference | None = None
    ):

        if reference is None:
            reference = GaussianReference()
        elif not isinstance(reference, GaussianReference):
            msg = "Reference density should be Gaussian."
            raise Exception(msg)

        L: Tensor = linalg.cholesky(cov)
        R: Tensor = linalg.inv(L)
        dim = mean.numel()

        def _check_subset(subset: str) -> None:
            if subset == "last":
                msg = "Preconditioner is only well-defined when subset=='first'"
                raise Exception(msg)
            return

        def Q(us: Tensor, subset: str) -> Tensor:
            _check_subset(subset)
            d_us = us.shape[1]
            xs = mean[:d_us] + (us @ L[:d_us, :d_us].T)
            return xs
        
        def Q_inv(xs: Tensor, subset: str) -> Tensor:
            _check_subset(subset)
            d_xs = xs.shape[1]
            us = (xs - mean[:d_xs]) @ R[:d_xs, :d_xs].T
            return us
        
        def neglogdet_Q(us: Tensor, subset: str) -> Tensor:
            _check_subset(subset)
            d_us = us.shape[1]
            neglogdets = -L.diag()[:d_us].log().sum()
            return neglogdets 
        
        def neglogdet_Q_inv(xs: Tensor, subset: str) -> Tensor: 
            _check_subset(subset)
            d_xs = xs.shape[1]
            neglogdets = -R.diag()[:d_xs].log().sum()
            return neglogdets 
        
        Preconditioner.__init__(
            self, 
            reference=reference, 
            Q=Q,
            Q_inv=Q_inv,
            neglogdet_Q=neglogdet_Q,
            neglogdet_Q_inv=neglogdet_Q_inv,
            dim=dim
        )
        return