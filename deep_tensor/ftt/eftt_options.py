from dataclasses import dataclass

from ..verification import verify_method


FIBRE_METHODS = ["aca", "random"]


@dataclass
class EFTTOptions():
    r"""Options for configuring the construction of an EFTT object.
    
    Parameters
    ----------
    num_error_samples:
        The number of samples to use when estimating the L2 error of 
        the FTT approximation to the target function at each iteration.
    fibre_method:
        The method used to compute a set of mode-$k$ fibres in each 
        dimension $k \in \{1, \dots, d\}$. This can be `"aca"` 
        (apply adaptive cross approximation as in @Strossner2024), or 
        `"random"` (choose a set of fibres at random).
    tol_svd: 
        The threshold to use when applying truncated SVD to compute an
        approximate basis for the mode-$k$ fibres in each dimension. 
        The minimum number of singular values such that the sum of 
        their squares exceeds ($1-$ `tol_svd`) will be retained.
    num_aca: 
        If `fibre_method="aca"`, the number of elements of the fibre 
        matrix to sample at each iteration when selecting a new pivot 
        element.
    tol_aca: 
        If `fibre_method="aca"`, the stopping tolerance, $\eps$, to 
        use. More concretely, if $\mathcal{S}$ denotes a set of 
        randomly-sampled elements of the mode-$k$ fibre matrix 
        $\boldsymbol{M}$ (and $\boldsymbol{I}$ and $\boldsymbol{J}$ 
        denote the current sets of row and column indices), the 
        iteration is considered finished when
        $$
            \max_{(i, j) \in \mathcal{S}}|A_{ij}| < \eps,
        $$
        where 
        $$
            \boldsymbol{A} = 
                \boldsymbol{M} - \boldsymbol{M}[:, \boldsymbol{J}]
                    \mathcal{M}[\boldsymbol{I}, \boldsymbol{J}]^{-1}
                    \mathcal{M}[\boldsymbol{I}, :].
        $$
    max_tucker_rank:
        If `fibre_method="aca"`, the maximum number of fibres to 
        generate.
    num_snapshots:
        If `fibre_method="snapshots"`, the number of snapshots to 
        sample.
    
    """
        
    num_error_samples: int = 1000
    fibre_method: str = "random"
    tol_svd: float = 1e-12
    num_aca: int = 50
    tol_aca: float = 1e-10
    max_tucker_rank: int = 30
    num_snapshots: int = 30
    
    def __post_init__(self):
        verify_method(self.fibre_method, FIBRE_METHODS)
        return