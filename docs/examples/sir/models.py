from numpy import ndarray
import torch 
from torch import Tensor 
from torchdiffeq import odeint


class SIRModel():
    """An SIR model with a single compartment. Setup replicated from 
    Cui, Dolgov and Zahm (2023).
    """

    def __init__(
        self, 
        S0: float = 99.0,
        I0: float = 1.0,
        R0: float = 0.0,
        t_eval: Tensor | None = None
    ):

        if t_eval is None:
            t_eval = torch.tensor([0.0, 1.25, 2.5, 3.75, 5.0]) 
        
        self.S0 = S0 
        self.I0 = I0 
        self.R0 = R0
        self.y0 = torch.tensor([S0, I0, R0])
        self.t_eval = t_eval
        return
    
    def _solve(self, params: Tensor) -> Tensor:
        """Solves the forward problem with given parameters, and 
        returns the number of infected people at the desired times.
        """

        b, g = params.T
        num_params = params.shape[0]

        def sir_func(t, y: ndarray) -> Tensor:
            S, I, _ = y.reshape(3, -1)
            return torch.vstack([-b*S*I, b*S*I - g*I, g*I]).flatten()
        
        y0 = self.y0.repeat_interleave(num_params)
        sol = odeint(sir_func, y0, self.t_eval, rtol=1e-06, atol=1e-06)
        return sol.T[num_params:(2*num_params), 1:]  # type: ignore
    
    def solve(self, params: Tensor, batch_size: int = 100_000) -> Tensor:
        # For an unknown reason, the ODE solver fails with a 
        # segmentation fault if the dimension of the state is quite 
        # large.
        param_batches = [params[i:i+batch_size] 
                         for i in range(0, params.shape[0], batch_size)]
        ys = torch.vstack([self._solve(p) for p in param_batches])        
        return ys


class SIRCompartmentModel():
    """An SIR model with multiple compartments. Setup replicated from 
    Cui, Dolgov and Scheichl (2024).
    """

    def __init__(
        self, 
        adjacency_matrix: Tensor,
        S0: Tensor,
        I0: Tensor,
        R0: Tensor,
        t_eval: Tensor,
        inds_obs: Tensor
    ):
        
        self.A = adjacency_matrix
        self.K = self.A.shape[0]

        self.S0 = S0 
        self.I0 = I0 
        self.R0 = R0
        self.y0 = torch.vstack([S0, I0, R0])
        
        self.t_eval = t_eval
        self.inds_obs = inds_obs
        self.num_timesteps = len(t_eval)
        self.num_obs = len(inds_obs) - 1  # Exclude initial timestep
        
        return

    def _get_infected(self, y: Tensor, num_params: int) -> Tensor:
        infected = y[:, (num_params*self.K):(2*num_params*self.K)].T
        infected = infected.reshape(num_params, self.K * self.num_timesteps)
        return infected
    
    def _solve(self, params: Tensor) -> Tensor:
        """Solves the forward problem with given parameters, and 
        returns the number of infected people at each of the output 
        times.
        """

        num_params = params.shape[0]
        thetas = params[:, 0::2]
        nus = params[:, 1::2]

        def sir_func(t, y: Tensor) -> Tensor:

            S, I, R = y.reshape(3, num_params, self.K)

            S_neighbours = (self.A[None, ...] * (S[..., None] - S[:, None, :])).sum(dim=1)
            I_neighbours = (self.A[None, ...] * (I[..., None] - I[:, None, :])).sum(dim=1)
            R_neighbours = (self.A[None, ...] * (R[..., None] - R[:, None, :])).sum(dim=1)
            
            dSdt = -thetas * S * I + 0.5 * S_neighbours
            dIdt = thetas * S * I - nus * I + 0.5 * I_neighbours
            dRdt = nus * I + 0.5 * R_neighbours
            
            return torch.vstack([dSdt, dIdt, dRdt]).flatten()
        
        y0 = torch.tile(self.y0, (1, num_params)).flatten()
        sol = odeint(sir_func, y0, self.t_eval, rtol=1e-6, atol=1e-6)

        ys = self._get_infected(sol, num_params)  # type: ignore
        return ys
    
    def solve(self, params: Tensor, batch_size: int = 100_000) -> Tensor:
        param_batches = [params[i:i+batch_size] 
                         for i in range(0, params.shape[0], batch_size)]
        ys = torch.vstack([self._solve(p) for p in param_batches])
        return ys
    
    def get_obs(self, infected: Tensor) -> Tensor:
        """Given a set of output from self.solve(), returns the 
        proportion of infected people in each compartment at the 
        observation times, for each set of parameters. 
        """
        infected = infected.reshape(-1, self.num_timesteps)
        obs = infected[:, self.inds_obs[1:]]
        return obs.reshape(-1, self.K*self.num_obs)
    
    def response_func(self, infected: Tensor) -> Tensor:
        """Given a set of output from self.solve(), returns the maximum 
        proportion of infected people in the final compartment over the 
        time horizon for each set of parameters.
        """
        infected = infected.reshape(-1, self.num_timesteps)
        infected_last_compartment = infected[(self.K-1)::self.K]
        return torch.max(infected_last_compartment, dim=1).values