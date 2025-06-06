from typing import Tuple

import torch
from torch import Tensor

from .marginal_estimator import MarginalEstimator
from ..references import Reference


class HistogramEstimator(MarginalEstimator):
    """Generates a histogram-based estimator of the (one-dimensional)
    density associated with a set of samples.

    Parameters
    ----------
    data:
        An n-dimensional vector containing a set of samples.
    n_bins:
        The number of (equisized) bins to divide the data into when 
        constructing the histogram. If not specified, a value of 
        ceil(sqrt(n)) will be used.
    defensive:
        Constant to add to ensure PDF is not equal to 0 anywhere.

    """

    def __init__(
        self, 
        data: Tensor,
        reference: Reference,
        bounds: Tensor | None = None,
        n_bins: Tensor | int | None = None, 
        defensive: Tensor | float = 1e-08
    ):

        if not isinstance(data, Tensor): 
            data = torch.tensor(data)

        lb, ub = data.min(), data.max()
        dx = ub - lb
        if bounds is None:
            bounds = torch.tensor([lb - 0.01*dx, ub + 0.01*dx])

        self.data = data
        self.reference = reference
        self.data_bounds = torch.tensor([lb, ub])
        self.bounds = bounds

        bounds_ref = reference.domain.bounds
        range_ref = bounds_ref[1] - bounds_ref[0]
        self.a = 0.5 * (self.bounds[0] + self.bounds[1])
        self.b = (self.bounds[1] - self.bounds[0]) / range_ref

        self._compute_parameters_tail()

        if n_bins is None:
            n_bins = torch.tensor(data.numel()).sqrt()
            n_bins = int(torch.ceil(n_bins))

        self.n = self.data.numel()
        self.n_bins = n_bins

        self.edges = torch.linspace(*self.data_bounds, self.n_bins + 1)
        self.counts = torch.histc(self.data, self.n_bins, *self.data_bounds)
        self.binwidth = (self.data_bounds[1] - self.data_bounds[0]) / self.n_bins

        self.pdf_centre = self.counts + defensive 
        self.pdf_centre = self.mass_centre * self.pdf_centre / (self.pdf_centre * self.binwidth).sum()

        self.cdf_centre = torch.tensor([0.0, *self.pdf_centre]).cumsum(dim=0) / self.pdf_centre.sum()
        self.cdf_centre = self.cdf_lhs + self.mass_centre * self.cdf_centre

        self.log2pi = torch.tensor(2.0*torch.pi).log()

        return
    
    def reference2hist(self, rs: Tensor) -> Tensor:
        return self.a + self.b * rs
    
    def hist2reference(self, xs: Tensor) -> Tensor:
        return (xs - self.a) / self.b 
    
    def pdf_tail(self, xs: Tensor) -> Tensor:
        rs = self.hist2reference(xs)
        return self.reference.eval_pdf(rs)[0] / self.b
    
    def cdf_tail(self, xs: Tensor) -> Tensor:
        rs = self.hist2reference(xs)
        return self.reference.eval_cdf(rs)[0]
    
    def _compute_parameters_tail(self) -> None:
        self.cdf_lhs, self.cdf_rhs = self.cdf_tail(self.data_bounds)
        self.mass_centre = self.cdf_rhs - self.cdf_lhs
        return

    def _eval_cdf_tails(self, xs: Tensor) -> Tensor:
        return self.cdf_tail(xs)
    
    def _eval_cdf_centre(self, xs: Tensor) -> Tensor:

        inds_left = torch.sum(xs[:, None] > self.edges, dim=1) - 1
        inds_left = inds_left.clamp(0, self.pdf_centre.numel()-1)
        us = self.cdf_centre[inds_left] + self.pdf_centre[inds_left] * (xs - self.edges[inds_left])
        return us

    def eval_cdf(self, xs: Tensor) -> Tuple[Tensor, Tensor]:

        cdfs = torch.zeros_like(xs)

        mask_tails = torch.bitwise_or(xs < self.data_bounds[0], xs > self.data_bounds[1])
        mask_centre = ~mask_tails

        cdfs[mask_tails] = self._eval_cdf_tails(xs[mask_tails])
        cdfs[mask_centre] = self._eval_cdf_centre(xs[mask_centre])
        pdfs = self.eval_pdf(xs)
        return cdfs, pdfs

    def _invert_cdf_tails(self, zs: Tensor) -> Tensor:
        rs = self.reference.invert_cdf(zs)
        xs = self.reference2hist(rs)
        return xs
    
    def _invert_cdf_centre(self, zs: Tensor) -> Tensor:
        inds_left = torch.sum(zs[:, None] > self.cdf_centre, dim=1) - 1
        inds_left = inds_left.clamp(0, self.pdf_centre.numel() - 1)
        xs = self.edges[inds_left] + (zs - self.cdf_centre[inds_left]) / self.pdf_centre[inds_left]
        return xs
    
    def invert_cdf(self, zs: Tensor) -> Tensor:

        xs = torch.zeros_like(zs)

        mask_tails = torch.bitwise_or(zs < self.cdf_lhs, zs > self.cdf_rhs)
        mask_centre = ~mask_tails

        xs[mask_tails] = self._invert_cdf_tails(zs[mask_tails])
        xs[mask_centre] = self._invert_cdf_centre(zs[mask_centre])
        
        return xs
    
    def _eval_pdf_tails(self, xs: Tensor) -> Tensor:
        return self.pdf_tail(xs)

    def _eval_pdf_centre(self, xs: Tensor) -> Tensor:
        inds_left = torch.sum(xs[:, None] > self.edges, dim=1) - 1
        inds_left = inds_left.clamp(0, self.pdf_centre.numel() - 1)
        return self.pdf_centre[inds_left]

    def eval_pdf(self, xs: Tensor) -> Tensor:

        pdfs = torch.zeros_like(xs)

        mask_tails = torch.bitwise_or(xs < self.data_bounds[0], xs > self.data_bounds[1])
        mask_centre = ~mask_tails

        pdfs[mask_tails] = self._eval_pdf_tails(xs[mask_tails])
        pdfs[mask_centre] = self._eval_pdf_centre(xs[mask_centre])
        return pdfs
    
    def eval_potential(self, xs: Tensor) -> Tensor:
        return -self.eval_pdf(xs).log()