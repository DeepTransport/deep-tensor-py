from dataclasses import dataclass

from ..verification import verify_method


FIBRE_METHODS = ["aca", "random"]


@dataclass
class EFTTOptions():
    r"""Options for configuring the construction of an EFTT object.
    
    Parameters
    ----------
    num_error_samples:
        The number of samples to use when estimating the $L^{2}$ error 
        of the FTT approximation to the target function at each 
        iteration.
    fibre_method:
        The method used to compute a set of mode-$k$ fibres in each 
        dimension $k \in \{1, \dots, d\}$. This can be `"aca"` 
        [apply adaptive cross approximation, as in @Strossner2024], or 
        `"random"` (choose a set of fibres at random).
    tol_svd: 
        The threshold to use when applying truncated SVD to compute an
        (approximate) orthogonal basis for the mode-$k$ fibres in each 
        dimension. The minimum number of singular values such that 
        their sum exceeds ($1-$ `tol_svd`) will be retained.
    num_aca: 
        If `fibre_method="aca"`, the number of elements of the fibre 
        matrix to sample at each iteration when selecting a new pivot 
        element.
    tol_aca: 
        If `fibre_method="aca"`, the stopping tolerance, $\epsilon$, to 
        use. More concretely, if $\mathcal{S}$ denotes a set of 
        randomly-sampled indices of the mode-$k$ fibre matrix 
        $\boldsymbol{M}$ (and $\mathcal{I}$ and $\mathcal{J}$ 
        denote the current sets of row and column indices), the 
        iteration is considered finished when
        $$
            \max_{(i, j) \in \mathcal{S}}|R_{ij}| < \epsilon,
        $$
        where the *residual* matrix $\boldsymbol{R}$ is given by
        $$
            \boldsymbol{R} = 
                \boldsymbol{M} - \boldsymbol{M}[:, \mathcal{J}]
                    \boldsymbol{M}[\mathcal{I}, \mathcal{J}]^{-1}
                    \boldsymbol{M}[\mathcal{I}, :].
        $$
    max_fibres:
        If `fibre_method="aca"`, the maximum number of fibres to 
        sample.
    num_snapshots:
        If `fibre_method="random"`, the number of snapshots to 
        sample.
    
    """
        
    num_error_samples: int = 1000
    fibre_method: str = "random"
    tol_svd: float = 1e-12
    num_aca: int = 50
    tol_aca: float = 1e-10
    max_fibres: int = 30
    num_snapshots: int = 30
    
    def __post_init__(self):
        verify_method(self.fibre_method, FIBRE_METHODS)
        return