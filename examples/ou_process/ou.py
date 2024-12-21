import torch


class OU():

    def __init__(self, d: float, a: float):

        self.a = torch.tensor(a)
        self.d = torch.tensor(d)

        self.A = torch.diag(torch.full((self.d-1,), -torch.sqrt(1.0 - self.a**2)), -1) + torch.eye(self.d)
        self.D = torch.diag(torch.concatenate((torch.tensor([1]), torch.full((d-1,), a))))

        self.B = torch.linalg.solve(self.D, self.A)
        self.Q = self.B.T @ self.B
        self.C = torch.linalg.inv(self.Q)

        self.norm = torch.sqrt((2.0*torch.pi)**self.d / torch.linalg.det(self.Q))

        return

    def eval_potential(self, x: torch.Tensor) -> torch.Tensor:
        f = 0.5 * torch.sum((self.B @ x.T) ** 2, 0) + torch.log(self.norm)
        return f