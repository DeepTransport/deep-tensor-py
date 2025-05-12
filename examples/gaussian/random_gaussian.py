import torch
from torch import Tensor
from torch import linalg
from torch.distributions import Exponential, MultivariateNormal


class RandomGaussian(object):

    def __init__(
        self, 
        dim: int = 20, 
        n_obs: int = 10, 
        sigma_e: float = 1.0
    ):

        self.dim = dim
        self.n_obs = n_obs

        self.mu_pri = self._generate_random_mu()
        self.cov_pri, self.cov_pri_inv = self._generate_random_cov()
        self.L_pri: Tensor = linalg.cholesky(self.cov_pri)
        self.R_pri: Tensor = linalg.inv(self.L_pri)
        self.det_R = torch.logdet(self.R_pri)

        self.G = self._generate_random_mapping()

        # Define properties of error
        self.mu_e = torch.zeros(self.n_obs)
        self.cov_e = sigma_e ** 2 * torch.eye(self.n_obs)
        self.R_e: Tensor = linalg.inv(linalg.cholesky(self.cov_e))

        # Sample true parameter from prior
        xi_true = torch.normal(mean=0.0, std=1.0, size=(self.dim,))
        self.m_true = self.mu_pri + self.L_pri @ xi_true

        # Generate synthetic data
        self.y_true = self.G @ self.m_true
        self.error = MultivariateNormal(self.mu_e, self.cov_e).sample()
        self.d_obs = self.y_true + self.error

        # Define true posterior
        H: Tensor = linalg.inv(self.G @ self.cov_pri @ self.G.T + self.cov_e)
        self.mu_post = self.mu_pri + self.cov_pri @ self.G.T @ H @ (self.d_obs - self.G @ self.mu_pri)
        self.cov_post = self.cov_pri - self.cov_pri @ self.G.T @ H @ self.G @ self.cov_pri
        self.R_post: Tensor = linalg.inv(linalg.cholesky(self.cov_post))
        
    def _generate_random_mu(self) -> Tensor:
        """Generates a random mean vector."""
        mu = torch.normal(mean=0.0, std=2.0, size=(self.dim,))
        return mu
    
    def _generate_random_cov(self) -> Tensor:
        """Generates a random covariance matrix."""
        
        # Generate random set of eigenvalues
        s = Exponential(rate=0.1).sample((self.dim, ))
        s = s.sort(descending=True)[0] + 1e-4

        # Generate random set of eigenvectors
        P = torch.normal(mean=0.0, std=1.0, size=(self.dim, self.dim))
        norms = torch.linalg.norm(P, axis=0, keepdims=True)
        P /= norms

        # Generate covariance and inverse of covariance
        cov = P @ torch.diag(s) @ P.T
        cov_inv = P @ torch.diag(1.0 / s) @ P.T

        return cov, cov_inv
    
    def _generate_random_mapping(self) -> Tensor:
        """Generates a random linear mapping from parameters to 
        observations.
        """
        G = torch.normal(mean=0.0, std=1.0, size=(self.n_obs, self.dim))
        return G
    
    def Q(self, xs: Tensor) -> Tensor:
        d_xs = xs.shape[1]
        return self.mu_pri[:d_xs] + xs @ self.L_pri[:d_xs, :d_xs].T

    def Q_inv(self, ms: Tensor) -> Tensor:
        d_ms = ms.shape[1]
        return (ms - self.mu_pri[:d_ms]) @ self.R_pri[:d_ms, :d_ms].T

    def neglogdet_Q(self, xs: Tensor) -> Tensor:
        n_xs, d_xs = xs.shape 
        return torch.full((n_xs,), -self.L_pri[:d_xs, :d_xs].diag().log().sum())

    def neglogdet_Q_inv(self, ms: Tensor) -> Tensor:
        n_ms, d_ms = ms.shape
        return torch.full((n_ms,), -self.R_pri[:d_ms, :d_ms].diag().log().sum())
    
    def neglogpri(self, ms: Tensor) -> Tensor:
        return 0.5 * ((ms - self.mu_pri) @ self.R_pri.T).square().sum(dim=1)
    
    def negloglik(self, ms: Tensor) -> Tensor:
        return 0.5 * ((ms @ self.G.T - self.d_obs) @ self.R_e).square().sum(dim=1)
    
    def potential_joint(self, ms: Tensor) -> Tensor:
        return (0.5 * self.dim * torch.tensor(2.0*torch.pi).log() 
                + 0.5 * self.cov_post.logdet()
                + 0.5 * ((ms - self.mu_post) @ self.R_post.T).square().sum(dim=1))
    
    def potential_marg(self, ms: Tensor) -> Tensor:
        dim_m = ms.shape[1]
        return (0.5 * dim_m * torch.tensor(2.0*torch.pi).log() 
                + 0.5 * self.cov_post[:dim_m, :dim_m].logdet()
                + 0.5 * ((ms - self.mu_post[:dim_m]) @ self.R_post[:dim_m, :dim_m].T).square().sum(dim=1))
    
    def potential_cond(self, mx_cond: Tensor, mys: Tensor) -> Tensor:

        dim_mx = mx_cond.shape[1]
        dim_my = mys.shape[1]

        mu_cond = self.mu_post[dim_mx:] + self.cov_post[dim_mx:, :dim_mx] @ linalg.inv(self.cov_post[:dim_mx, :dim_mx]) @ (mx_cond.flatten() - self.mu_post[:dim_mx])
        # mu_cond = mu_cond.flatten()
        cov_cond = self.cov_post[dim_mx:, dim_mx:] - self.cov_post[dim_mx:, :dim_mx] @ linalg.inv(self.cov_post[:dim_mx, :dim_mx]) @ self.cov_post[:dim_mx, dim_mx:]
        R_cond: Tensor = linalg.inv(linalg.cholesky(cov_cond))

        print(mu_cond)
        print(cov_cond)

        return (0.5 * dim_my * torch.tensor(2.0*torch.pi).log() 
                + 0.5 * cov_cond.logdet()
                + 0.5 * ((mys - mu_cond) @ R_cond.T).square().sum(dim=1))