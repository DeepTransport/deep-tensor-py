from torch import Tensor

from .preconditioner import Preconditioner
# from ..irt import DIRT


MAPPING2SUBSET = {"lower": "first", "upper": "last"}


class DIRTPreconditioner(Preconditioner):
    r"""A preconditioner built using a previously constructed DIRT 
    object.
    
    Parameters
    ----------
    dirt: DIRT
        A previously constructed DIRT object.
    mapping:
        Whether the transformations associated with the DIRT object are 
        lower or upper triangular.

    """

    def __init__(self, dirt, mapping: str = "lower"):

        try:
            subset = MAPPING2SUBSET[mapping.lower()]
        except:
            msg = f"Unknown mapping encountered: {mapping}."
            raise Exception(msg)
        
        def Q(xs: Tensor) -> Tensor:
            return dirt.eval_irt(xs, subset=subset)[0]

        def Q_inv(ms: Tensor) -> Tensor:
            return dirt.eval_rt(ms, subset=subset)[0]
        
        def neglogdet_Q(xs: Tensor):
            neglogfxs = dirt.eval_irt(xs, subset=subset)[1]
            neglogrefs = dirt.reference.eval_potential(xs)[0]
            return neglogrefs - neglogfxs

        def neglogdet_Q_inv(ms: Tensor) -> Tensor:
            xs, neglogfxs = dirt.eval_rt(ms, subset=subset)
            neglogrefs = dirt.reference.eval_potential(xs)[0]
            return neglogfxs - neglogrefs

        Preconditioner.__init__(
            self, 
            reference=dirt.reference, 
            Q=Q, 
            Q_inv=Q_inv, 
            neglogdet_Q=neglogdet_Q, 
            neglogdet_Q_inv=neglogdet_Q_inv,
            dim=dirt.dim
        )

        return