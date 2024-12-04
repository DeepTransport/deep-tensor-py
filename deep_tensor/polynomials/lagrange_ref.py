import torch

from .jacobi_11 import Jacobi11


class LagrangeRef():
    
    def __init__(self, n: int):
        """Defines the reference Lagrange basis, in the reference
        domain [0, 1].
        """

        if n < 2: 
            msg = ("More than two points are needed " 
                   + "to define Lagrange interpolation.")
            raise Exception(msg)
        
        self.nodes = torch.zeros(n)
        self.nodes[-1] = 1.0

        if n > 2:
            order = n-3
            jacobi = Jacobi11(order)
            self.nodes[1:-1] = 0.5 * (jacobi.nodes + 1.0)

        self.omega = torch.zeros(n)
        for j in range(n):
            mask = torch.full((n, ), True)
            mask[j] = False
            self.omega[j] = 1.0 / torch.prod(self.nodes[j]-self.nodes[mask])
        
        
            

