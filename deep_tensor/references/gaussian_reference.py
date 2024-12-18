import torch

from .symmetric_reference import SymmetricReference


class GaussianReference(SymmetricReference):
    """The Gaussian reference distribution."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def eval_ref_cdf(self, z: torch.Tensor) -> torch.Tensor:
        u = 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))
        f = torch.exp(-0.5 * z**2) / torch.sqrt(torch.tensor(2.0 * torch.pi))
        return u, f
    
    def eval_ref_pdf(self, z: torch.Tensor) -> torch.Tensor:
        f = torch.exp(-0.5 * z**2) / torch.sqrt(torch.tensor(2.0 * torch.pi))
        g = -z * f 
        return f, g
    
    def invert_ref_cdf(self, u: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2.0*u-1.0)

    def log_joint_ref_pdf(self, rs: torch.Tensor) -> torch.Tensor:
        num_r = rs.shape[1]
        frs = (-0.5 * num_r * torch.log(torch.tensor(2.0*torch.pi)) 
             + torch.sum(-0.5 * rs**2, dim=1))
        grs = -rs
        return frs, grs