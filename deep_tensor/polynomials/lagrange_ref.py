import torch


class LagrangeRef():
    
    def __init__(self, n: int):

        if n < 2: 
            msg = ("More than two points are needed " 
                   + "to define Lagrange interpolation.")
            raise Exception(msg)
        
        self.nodes = torch.zeros(n)
        self.nodes[-1] = 1.0

        if n > 2:
            order = n-3
            

