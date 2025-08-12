import warnings

import numpy as np
from numpy import ndarray
import torch 
from torch import Tensor 
from scipy.integrate import solve_ivp


class SIRModel():

    def __init__(
        self, 
        adjacency_mat: np.ndarray,
        S0: np.ndarray,
        I0: np.ndarray,
        R0: np.ndarray,
        t1: float,
        t_eval: ndarray
    ):
        """A simple SIR model. Setup replicated from Cui, Dolgov and 
        Scheichl (2023).
        """
        
        self.A = adjacency_mat
        self.K = self.A.shape[0]

        self.S0 = S0 
        self.I0 = I0 
        self.R0 = R0
        self.y0 = np.array([S0, I0, R0])
        
        self.t_span = (0.0, t1)
        self.t_eval = t_eval
        
        return
    
    @staticmethod 
    def sir_func(t, y: ndarray, A: np.ndarray, params: np.ndarray) -> ndarray:
    
        num_compartments = A.shape[0]
        num_params = params.shape[0]

        thetas = params[:, 0::2]
        nus = params[:, 1::2]

        S, I, R = y.reshape(3, num_params, num_compartments)

        S_neighbours = (A[None, ...] * (S[..., None] - S[:, None, :])).sum(axis=1)
        I_neighbours = (A[None, ...] * (I[..., None] - I[:, None, :])).sum(axis=1)
        R_neighbours = (A[None, ...] * (R[..., None] - R[:, None, :])).sum(axis=1)
        
        dSdt = -thetas * S * I + 0.5 * S_neighbours
        dIdt = thetas * S * I - nus * I + 0.5 * I_neighbours
        dRdt = nus * I + 0.5 * R_neighbours
        
        return np.array([dSdt, dIdt, dRdt]).flatten()
    
    def _extract_obs(self, y: np.ndarray, num_params: int) -> np.ndarray:

        I = y[(num_params*self.K):(2*num_params*self.K)]
        I = I.reshape(num_params, -1)
        return I
    
    def _solve_fwd(self, params: Tensor) -> np.ndarray:
        """Solves the forward problem with given parameters, and 
        returns the number of infected people at the desired times.
        """

        num_params = params.shape[0]
        y0 = np.tile(self.y0, (1, num_params)).flatten()

        sol = solve_ivp(
            fun=self.sir_func, 
            t_span=self.t_span,
            y0=y0, 
            args=(self.A, params.numpy()), 
            t_eval=self.t_eval,
            rtol=1e-6, 
            atol=1e-6
        )
        if not sol.success:
            msg = "Forward solver did not converge."
            warnings.warn(msg)
        
        ys = self._extract_obs(sol.y, num_params)
        return ys
    
    def solve_fwd(self, params: Tensor, num_batches: int = 10_000) -> Tensor:
        # For an unknown reason, the ODE solver fails with a 
        # segmentation fault if the dimension of the state is quite 
        # large.
        param_batches = [params[i:i+num_batches] for i in range(0, params.shape[0], num_batches)]
        ys = np.vstack([self._solve_fwd(p) for p in param_batches])
        ys = torch.from_numpy(ys).to(dtype=torch.get_default_dtype())
        return ys
    

if __name__ == "__main__":

    num_compartments = 4
    dim = 2 * num_compartments

    adjacency_mat = np.zeros((num_compartments, num_compartments))
    for i in range(num_compartments):
        adjacency_mat[i][i] = 1
        adjacency_mat[i][(i-1)%num_compartments] = 1
        adjacency_mat[i][(i+1)%num_compartments] = 1

    S0 = np.arange(100-num_compartments, 100)
    I0 = 100 - S0
    R0 = np.zeros((num_compartments,))

    t1 = 5.0
    t_eval = (5.0 / 6.0) * np.arange(1, 7)

    solver = SIRModel(adjacency_mat, S0, I0, R0, t1, t_eval)

    true_param = torch.tensor([[0.1, 1.0] for _ in range(num_compartments)]).reshape(1, -1)

    # params = torch.rand((100_000, dim))
    # params[1:] = params[0]

    import time 
    t0 = time.time()
    ys = solver.solve_fwd(true_param)
    print(ys)
    t1 = time.time()
    print(t1-t0)