import warnings

import numpy as np
from numpy import ndarray
import torch 
from torch import Tensor 
from scipy.integrate import solve_ivp


class SIRModel():

    def __init__(
        self, 
        S0: float = 99.0,
        I0: float = 1.0,
        R0: float = 0.0,
        t1: float = 5.0,
        t_eval: Tensor | ndarray | None = None
    ):
        """A SIR model with a single compartment. Setup replicated from 
        Cui, Dolgov and Zahm (2023).
        """

        if t_eval is None:
            t_eval = np.array([1.25, 2.5, 3.75, 5.0]) 
        
        self.S0 = S0 
        self.I0 = I0 
        self.R0 = R0
        self.y0 = np.array([S0, I0, R0])
        self.t_span = (0.0, t1)
        self.t_eval = np.array(t_eval)
        
        return
    
    @staticmethod 
    def sir_func(t, y: ndarray, b: float, g: float) -> ndarray:
        S, I, _ = y.reshape(3, -1)
        return np.array([-b*S*I, b*S*I - g*I, g*I]).flatten()
    
    def _solve(self, params: Tensor) -> ndarray:
        """Solves the forward problem with given parameters, and 
        returns the number of infected people at the desired times.
        """

        num_params = params.shape[0]
        
        sol = solve_ivp(
            fun=self.sir_func, 
            t_span=self.t_span,
            y0=self.y0.repeat(num_params), 
            args=params.T.numpy(), 
            t_eval=self.t_eval
        )
        if not sol.success:
            msg = "Forward solver did not converge."
            warnings.warn(msg)
        
        return sol.y[num_params:(2*num_params)]  # infected only
    
    def solve(self, params: Tensor, batch_size: int = 10_000) -> Tensor:
        # For an unknown reason, the ODE solver fails with a 
        # segmentation fault if the dimension of the state is quite 
        # large.
        param_batches = [params[i:i+batch_size] 
                         for i in range(0, params.shape[0], batch_size)]
        ys = np.vstack([self._solve(p) for p in param_batches])
        ys = torch.from_numpy(ys).to(dtype=torch.get_default_dtype())
        return ys


class SIRCompartmentModel():

    def __init__(
        self, 
        adjacency_mat: ndarray,
        S0: ndarray,
        I0: ndarray,
        R0: ndarray,
        t1: float,
        t_eval: ndarray,
        inds_obs: ndarray
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
        self.inds_obs = inds_obs
        self.num_timesteps = len(t_eval)
        self.num_obs = len(inds_obs)
        
        return
    
    @staticmethod 
    def sir_func(t, y: ndarray, A: ndarray, params: ndarray) -> ndarray:
    
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

    def _get_infected(self, y: ndarray, num_params: int) -> ndarray:
        infected = y[(num_params*self.K):(2*num_params*self.K)]
        infected = infected.reshape(num_params, self.K * self.num_timesteps)
        return infected
    
    def _solve(self, params: Tensor) -> ndarray:
        """Solves the forward problem with given parameters, and 
        returns the number of infected people at each of the output 
        times.
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

        ys = self._get_infected(sol.y, num_params)
        return ys
    
    def solve(self, params: Tensor, batch_size: int = 10_000) -> Tensor:
        # For an unknown reason, the ODE solver fails with a 
        # segmentation fault if the dimension of the state is quite 
        # large.
        param_batches = [params[i:i+batch_size] 
                         for i in range(0, params.shape[0], batch_size)]
        ys = np.vstack([self._solve(p) for p in param_batches])
        ys = torch.from_numpy(ys).to(dtype=torch.get_default_dtype())
        return ys
    
    def get_obs(self, infected: Tensor) -> Tensor:
        """Given a set of output from self.solve(), returns the 
        proportion of infected people in each compartment at the 
        observation times, for each set of parameters. 
        """
        infected = infected.reshape(-1, self.num_timesteps)
        obs = infected[:, self.inds_obs]
        return obs.reshape(-1, self.K*self.num_obs)
    
    def response_func(self, infected: Tensor) -> Tensor:
        """Given a set of output from self.solve(), returns the maximum 
        proportion of infected people in the final compartment over the 
        time horizon for each set of parameters.
        """
        infected = infected.reshape(-1, self.num_timesteps)
        infected_last_compartment = infected[(self.K-1)::self.K]
        return torch.max(infected_last_compartment, dim=1).values