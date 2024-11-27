import torch

from .lagrange_ref import LagrangeRef
from .piecewise import Piecewise


class LagrangeP(Piecewise):

    def __init__(self, order, num_elems):

        Piecewise.__init__(self, order, num_elems)

        if order == 1:
            msg = ("When `order=1`, Lagrange1 should be used " 
                   + "instead of LagrangeP.")
            raise Exception(msg)
        
        self.local = LagrangeRef(self.order + 1)

        # Set up global nodes
        num_nodes = self.num_elements * (self.local.cardinality - 1) + 1

        return