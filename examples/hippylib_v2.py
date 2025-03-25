"""Computes a Laplace approximation to the posterior associated with a 
Poisson problem.
"""

import dolfin as dl
import hippylib as hl
import numpy as np
from matplotlib import pyplot as plt
import ufl

np.random.seed(1)


class PoissonProblem():
    """Class for the Poisson problem. Should be able to
     - generate samples from the prior
     - solve Poisson problem with a given sample of the diffusion coefficient
     - compute Laplace approximation
    """

    def __init__(
        self, 
        nx: int = 64, 
        ny: int = 64
    ):
        """Class for a two-dimensional Poisson problem.

        The domain for the problem is [0, 1]^2.
        
        Parameters
        ----------
        nx: 
            Discretisation in the x direction.
        ny: 
            Discretisation in the y direction.

        """

        self.mesh = dl.UnitSquareMesh(nx, ny)

        # Define spaces for state, parameter and adjoint variables
        self.Vh2 = dl.FunctionSpace(self.mesh, "Lagrange", 2)
        self.Vh1 = dl.FunctionSpace(self.mesh, "Lagrange", 1)
        self.Vh = [self.Vh2, self.Vh1, self.Vh2]  # TODO: move this outside?

        # Define boundary conditions for the forward and adjoint problems
        self.u_bound = dl.Expression("x[0]", degree=1)  # equal to 0 on left side and 1 on right side
        self.u_bound_0 = dl.Constant(0.0)  # equal to 0 on left and right sides
        self.bc = dl.DirichletBC(self.Vh[hl.STATE], self.u_bound, self.u_boundary)
        self.bc_0 = dl.DirichletBC(self.Vh[hl.STATE], self.u_bound_0, self.u_boundary)

        # Define forcing term
        # self.f = dl.Constant(0.0)

        self.pde = hl.PDEVariationalProblem(
            self.Vh, 
            self.variational_form, 
            self.bc, 
            self.bc_0, 
            is_fwd_linear=True
        )

        # TODO: tidy this up
        gamma = 0.1
        delta = 0.5
        self.prior = hl.BiLaplacianPrior(self.Vh[hl.PARAMETER], gamma, delta, robin_bc=True)

        # Sample truth from prior
        self.m_true = self.sample_prior()

        # objs = [dl.Function(self.Vh[hl.PARAMETER], self.m_true), dl.Function(self.Vh[hl.PARAMETER], self.prior.mean)]
        # mytitles = ["True Parameter", "Prior mean"]
        # hl.nb.multi1_plot(objs, mytitles)
        # plt.show()

        x0s_obs = np.linspace(0.1, 0.9, 10)
        x1s_obs = np.linspace(0.1, 0.9, 10)
        xs_obs = np.array([[x0, x1] for x0 in x0s_obs for x1 in x1s_obs])
        self.misfit = hl.PointwiseStateObservation(self.Vh[hl.STATE], xs_obs)

        u_true = self.pde.generate_state()  # Vector in shape of state
        x = [u_true, self.m_true, None]
        self.pde.solveFwd(x[hl.STATE], x)

        self.misfit.B.mult(x[hl.STATE], self.misfit.d)  # compute Bu and save result to self.obs.d (data)
        rel_noise = 0.01  # sd of noise = 1% of max absolute state value
        max_obs = self.misfit.B.norm("linf")
        noise_std = rel_noise * max_obs
        hl.parRandom.normal_perturb(noise_std, self.misfit.d)  # Add Gaussian perturbation to data
        self.misfit.noise_variance = noise_std ** 2

        # vmax = max(u_true.max(), self.misfit.d.max())
        # vmin = min(u_true.min(), self.misfit.d.min())

        # plt.figure(figsize=(15,5))
        # hl.nb.plot(dl.Function(self.Vh[hl.STATE], u_true), mytitle="True State", subplot_loc=121, vmin=vmin, vmax=vmax)
        # hl.nb.plot_pts(xs_obs, self.misfit.d, mytitle="Observations", subplot_loc=122, vmin=vmin, vmax=vmax)
        # plt.show()

        self.model = hl.Model(self.pde, self.prior, self.misfit)

        # m0 = dl.interpolate(dl.Expression("sin(x[0])", degree=5), self.Vh[hl.PARAMETER])
        # _ = hl.modelVerify(self.model, m0.vector())

        # plt.figure(figsize=(15,5))
        # hl.nb.plot(dl.Function(self.Vh[hl.STATE], x[hl.STATE]), subplot_loc=121,mytitle="State")
        # hl.nb.plot(dl.Function(self.Vh[hl.PARAMETER], x[hl.PARAMETER]), subplot_loc=122,mytitle="Parameter")
        # plt.show()

        return

    @staticmethod
    def u_boundary(x: np.ndarray, on_boundary: bool) -> bool:
        """Returns True if a point is located on the left or right side 
        of the domain, and False otherwise.
        """
        return on_boundary and (x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS)
    
    @staticmethod
    def variational_form(
        u: ufl.Argument, 
        m: ufl.Coefficient, 
        p: ufl.Argument
    ) -> ufl.Form:  # TODO: check annotations
        """Returns the variational form of the PDE.
        
        Parameters
        ----------
        u:
            State variable.
        m: 
            Parameter.
        p: 
            Adjoint variable.

        Returns
        -------
        varf:
            The variational form of the PDE.
        
        """
        f = dl.Constant(0.0)  # TODO: figure out why I can't get rid of ufl.dx
        return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx - f * p * ufl.dx
    
    def sample_prior(self):
        """Generates a sample from the prior.
        
        TODO: allow for the generation of multiple samples.

        """
        
        # Generate vector of Gaussian white noise
        noise = dl.Vector()
        self.prior.init_vector(noise, dim="noise")
        hl.parRandom.normal(1.0, noise)

        # Transform white noise to the correpsonding sample from the prior
        sample = dl.Vector()
        self.prior.init_vector(sample, dim=0)
        self.prior.sample(noise, sample)

        return sample
    
    def compute_map(self):
        """TODO: write docstring."""
        
        m = self.prior.mean.copy()
        solver = hl.ReducedSpaceNewtonCG(self.model)
        solver.parameters["rel_tolerance"] = 1e-6
        solver.parameters["abs_tolerance"] = 1e-12
        solver.parameters["max_iter"] = 25
        solver.parameters["GN_iter"] = 5
        solver.parameters["globalization"] = "LS"
        solver.parameters["LS"]["c_armijo"] = 1e-4

        x = solver.solve([None, m, None])

        if not solver.converged:
            raise Exception("Solver failed to converge.")
        
        print(f"Termination reason: {solver.termination_reasons[solver.reason]}")
        print(f"Final gradient norm: {solver.final_grad_norm}")
        print(f"Final cost: {solver.final_cost}")
        return x
    
    def form_post(self, u: list):
        """TODO: write docstring."""

        self.model.setPointForHessianEvaluations(x, gauss_newton_approx=False)
        Hmisfit = hl.ReducedHessian(self.model, misfit_only=True)
        k = 50
        p = 20
        print( "Single/Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )

        Omega = hl.MultiVector(x[hl.PARAMETER], k+p)
        hl.parRandom.normal(1., Omega)
        lmbda, V = hl.doublePassG(Hmisfit, self.prior.R, self.prior.Rsolver, Omega, k)

        post = hl.GaussianLRPosterior(self.prior, lmbda, V)
        post.mean = x[hl.PARAMETER]

        return post

if __name__ == "__main__":
    
    prob = PoissonProblem()
    x = prob.compute_map()
    post = prob.form_post()