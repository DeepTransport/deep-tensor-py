import torch

from .chebyshev_2nd_unweighted import Chebyshev2ndUnweighted
from .legendre import Legendre
from .spectral_cdf import SpectralCDF
from ..constants import EPS


class BoundedPolyCDF(Chebyshev2ndUnweighted, SpectralCDF):

    def __init__(self, poly: Legendre, **kwargs):
        
        Chebyshev2ndUnweighted.__init__(self, order=2*poly.order)
        SpectralCDF.__init__(self, **kwargs)
        
        return

    def grid_measure(self, n: int) -> torch.Tensor:
        
        grid = torch.linspace(*self.domain, n)
        grid[0] = self.domain[0] - EPS
        grid[-1] = self.domain[-1] + EPS
        
        return grid
    
    def eval_int_basis(self, rs: torch.Tensor) -> torch.Tensor:
        
        thetas = self.x2theta(rs)
        basis_vals = (torch.cos(torch.outer(thetas, self.n+1)) 
                      * (self.normalising / (self.n+1)))
        
        return basis_vals
    
    def eval_int_basis_newton(
        self, 
        rs: torch.Tensor
    ) -> torch.Tuple[torch.Tensor]:
        
        thetas = self.x2theta(rs)
        basis_vals = (torch.cos(torch.outer(thetas, self.n+1)) 
                      * (self.normalising / (self.n+1)))
        deriv_vals = (torch.sin(torch.outer(self.n+1, thetas)) 
                      / (torch.sin(thetas) / self.normalising)).T
            
        # % deal with endpoints
        
        # mask = abs(x+1) < eps;
        # if sum(mask) > 0
        #     db(mask,:) = repmat(((obj.n+1).*(-1).^obj.n).*obj.normalising, sum(mask), 1);
        # end
        
        # mask = abs(x-1) < eps;
        # if sum(mask) > 0
        #     db(mask,:) = repmat((obj.n+1).*obj.normalising, sum(mask), 1);
        # end

        return basis_vals, deriv_vals