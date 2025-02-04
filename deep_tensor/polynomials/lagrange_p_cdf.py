from typing import Tuple

import torch

from .bounded_poly_cdf import BoundedPolyCDF
from .chebyshev_2nd_unweighted import Chebyshev2ndUnweighted
from .lagrange_p import LagrangeP
from .piecewise_cdf import PiecewiseCDF


class LagrangePCDF(LagrangeP, PiecewiseCDF):

    def __init__(self, poly: LagrangeP, **kwargs):

        LagrangeP.__init__(self, poly.order, poly.num_elems)
        PiecewiseCDF.__init__(self, **kwargs)

        # Local CDF poly
        self.cheby, self.cdf_basis2node = self.lag2cheby(poly)

        self.mass = None 
        self.mass_R = None 
        self.int_W = None

        n_nodes = self.num_elems * (self.cheby.cardinality - 1) + 1
        self.nodes = torch.zeros(n_nodes)
        for i in range(self.num_elems):
            inds = torch.arange(self.cheby.cardinality) + i * (self.cheby.cardinality - 1)
            self.nodes[inds] = self.cheby.nodes * self.elem_size + self.grid[i]

        if self.cheby.cardinality > 2:
            j = torch.arange(self.cheby.cardinality, n_nodes, self.cheby.cardinality-1)
            raise NotImplementedError()
            self.global2local = None

        else: 
            raise NotImplementedError()

        raise NotImplementedError()
    
    def lag2cheby(
        self,
        poly: LagrangeP
    ) -> Tuple[BoundedPolyCDF, torch.Tensor]:
        """Defines a data structure which maps Lagrange polynomials to 
        Chebyshev polynomials with preserved boundary values. 

        TODO: finish.
        """

        cheby = BoundedPolyCDF(poly)
        n_nodes = cheby.cardinality

        if n_nodes < 3:
            msg = "Must use more than three nodes."
            raise Exception(msg)

        tmp = Chebyshev2ndUnweighted(n_nodes-3)

        ref_nodes = torch.tensor([cheby.domain[0], *tmp.nodes, cheby.domain[1]])
        cheby.basis2node = cheby.eval_basis(ref_nodes)

        # L, U = torch.linalg.lu(cheby.basis2node)
        cheby.node2basis = torch.linalg.inv(cheby.basis2node) # TODO: tidy this up

        cheby.mass_R = None 
        cheby.int_W = None      # TODO: figure out what is going on here.

        cheby.nodes = 0.5 * (ref_nodes + 1) # Map into [0, 1]?
        cdf_basis2node = 0.5 * cheby.eval_int_basis(ref_nodes)

        return cheby, cdf_basis2node
    

    
