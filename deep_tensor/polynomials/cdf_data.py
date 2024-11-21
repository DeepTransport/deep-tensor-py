import torch


class CDFData():

    def __init__(
        self, 
        num_samples: int,
        poly_coef: torch.Tensor, 
        cdf_poly_grid: torch.Tensor, 
        poly_norm: torch.Tensor
    ):
        
        self.num_samples = num_samples 
        self.poly_coef = poly_coef
        self.cdf_poly_grid = cdf_poly_grid
        self.poly_norm = poly_norm
        return