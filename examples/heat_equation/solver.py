# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import numpy as np
import ufl

import hippylib as hl


class SpaceTimePointwiseStateObservation(hl.Misfit):
    
    def __init__(self, Vh,
                 observation_times,
                 targets,
                 d = None,
                 noise_variance=None):
        
        self.Vh = Vh
        self.observation_times = observation_times
        
        self.B = hl.assemblePointwiseObservation(self.Vh, targets)
        self.ntargets = targets
        
        if d is None:
            self.d = hl.TimeDependentVector(observation_times)
            self.d.initialize(self.B, 0)
        else:
            self.d = d
            
        self.noise_variance = noise_variance
        
        ## TEMP Vars
        self.u_snapshot = dl.Vector()
        self.Bu_snapshot = dl.Vector()
        self.d_snapshot  = dl.Vector()
        self.B.init_vector(self.u_snapshot, 1)
        self.B.init_vector(self.Bu_snapshot, 0)
        self.B.init_vector(self.d_snapshot, 0)
        
    def observe(self, x, obs):        
        obs.zero()
        
        for t in self.observation_times:
            x[hl.STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            obs.store(self.Bu_snapshot, t)
            
    def cost(self, x):
        c = 0
        for t in self.observation_times:
            x[hl.STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            self.d.retrieve(self.d_snapshot, t)
            self.Bu_snapshot.axpy(-1., self.d_snapshot)
            c += self.Bu_snapshot.inner(self.Bu_snapshot)
            
        return c/(2.*self.noise_variance)
    
    def grad(self, i, x, out):
        out.zero()
        if i == hl.STATE:
            for t in self.observation_times:
                x[hl.STATE].retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.d.retrieve(self.d_snapshot, t)
                self.Bu_snapshot.axpy(-1., self.d_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                out.store(self.u_snapshot, t)           
        else:
            raise NotImplementedError()
            
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass
    
    def apply_ij(self, i,j, direction, out):
        out.zero()
        if i == hl.STATE and j == hl.STATE:
            for t in self.observation_times:
                direction.retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                out.store(self.u_snapshot, t)
        else:
            raise NotImplementedError()
        return


class HeatSolver(object):
    
    def __init__(
        self, 
        mesh, 
        Vh: dl.FunctionSpace, 
        prior: hl.modeling._Prior, 
        misfit: SpaceTimePointwiseStateObservation,
        ts: np.ndarray
    ):
        
        self.mesh = mesh
        self.Vh = Vh
        self.prior = prior
        self.misfit = misfit

        self.dt = ts[1] - ts[0]  # NOTE: assumes equispaced times
        self.ts = ts

        self.u = dl.TrialFunction(self.Vh[hl.STATE])
        self.v = dl.TestFunction(self.Vh[hl.STATE])

        self.u0 = dl.interpolate(dl.Constant(0.0), self.Vh[hl.STATE]).vector()
        
        # Define forcing function
        sd = "-1/(2*std::pow(r, 2))"
        f0 = "(std::pow(x[0]-a0, 2) + std::pow(x[1]-a1, 2))"
        f1 = "(std::pow(x[0]-b0, 2) + std::pow(x[1]-b1, 2))"
        self.f = dl.Expression(
            f"c * (std::exp({sd} * {f0}) - std::exp({sd} * {f1}))",
            c=5.0*np.pi*1e-5,
            r=0.05,
            a0=0.5, a1=0.5,
            b0=2.5, b1=0.5,
            element=Vh[hl.STATE].ufl_element()
        )
        self.f = dl.interpolate(self.f, self.Vh[hl.STATE]).vector()

        # Assemble mass matrix
        self.M = dl.assemble(self.u * self.v * ufl.dx)
        
        # self.bc = dl.DirichletBC(
        #     self.Vh[hl.STATE], 
        #     dl.Constant(0.0), 
        #     self.dirichlet_boundary
        # )

        # self.bc_p = dl.DirichletBC(
        #     self.Vh[hl.STATE],
        #     dl.Constant(0.0),
        #     self.dirichlet_boundary
        # )

        # Part of model public API (??)
        self.gauss_newton_approx = True

        # k_true = self.sample_prior()
        # u_true = self.generate_vector(hl.STATE)
        # self.x = [u_true, k_true, None]

        return

    @staticmethod
    def dirichlet_boundary(
        x: np.ndarray, 
        on_boundary: bool
    ) -> bool:
        """Returns True if a point is on the Dirichlet boundary, and 
        False otherwise.
        """
        return on_boundary and ((np.abs(x[0]-0.0) < 1e-8) or (np.abs(x[0]-3.0) < 1e-8))

    def vec2func(self, vec: dl.Vector, component: int) -> dl.Function:
        """Converts a vector to a function.
        """
        k = dl.Function(self.Vh[component])
        k.vector().set_local(vec.get_local()[:])
        k.vector().apply("insert")
        return k

    def sample_prior(self) -> dl.Vector:
        """Returns a single function sampled from the prior.
        """
        # Generate vector of white noise
        noise = dl.Vector()
        self.prior.init_vector(x=noise, dim="noise")
        hl.parRandom.normal(sigma=1.0, out=noise)
        # Generate prior sample
        ks = dl.Vector()
        self.prior.init_vector(x=ks, dim=0)
        self.prior.sample(noise=noise, s=ks)
        return ks

    def generate_vector(self, component="ALL"):
        """Generates an empty vector (or set of vectors) of the 
        appropriate dimension.
        """

        if component == "ALL":
            u = hl.TimeDependentVector(self.ts)
            u.initialize(self.M, 0)
            m = dl.Vector()
            self.prior.init_vector(m,0)
            p = hl.TimeDependentVector(self.ts)
            p.initialize(self.M, 0)
            return [u, m, p]
        
        elif component == hl.STATE:
            # Make size of all state vectors = number of rows of mass matrix
            # (and set all entries equal to 0s)
            u = hl.TimeDependentVector(self.ts)
            u.initialize(M=self.M, dim=0)
            return u
        
        elif component == hl.PARAMETER:
            m = dl.Vector()
            self.prior.init_vector(x=m, dim=0)
            return m
        
        elif component == hl.ADJOINT:
            p = hl.TimeDependentVector(self.ts)
            p.initialize(M=self.M, dim=0)
            return p
        
        else:
            raise Exception("Unknown component.")
    
    def init_parameter(self, m):
        self.prior.init_vector(m, 0)
          
    def cost(self, x):
        """TODO: figure out what this is doing."""
        Rdx = dl.Vector()
        self.prior.init_vector(Rdx,0)
        dx = x[hl.PARAMETER] - self.prior.mean
        self.prior.R.mult(dx, Rdx)
        reg = 0.5 * Rdx.inner(dx)
        misfit = self.misfit.cost(x)
        return [reg+misfit, reg, misfit]
    
    def solveFwd(
        self, 
        out: hl.TimeDependentVector, 
        x
    ) -> None:
        """Solves the forward problem.

        Time is discretised using the backward Euler method.
        
        Parameters
        ----------
        out:
            A set of state vectors at each time snapshot. Each vector 
            is stored in out.data.
        x:
            A list of the state, parameter and adjoint vectors.
        
        """

        # Remove old data in state vector
        out.zero()

        # Initialise state vector and right-hand side
        u = dl.Vector()
        self.M.init_vector(u, 0)
        rhs = dl.Vector()
        self.M.init_vector(rhs, 0)

        # Initialise previous state
        u_prev = self.u0.copy()

        # Convert parameter to function
        k = self.vec2func(x[hl.PARAMETER], hl.PARAMETER)

        # Assemble stiffness matrix
        K = ufl.exp(k) * ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        K = dl.assemble(K)
                
        # Define LHS of variational form
        A = self.M + self.dt * K
        # self.bc.apply(A)
        solver = dl.LUSolver(A)
        
        for t in self.ts[1:]:

            self.M.mult(self.dt * self.f + u_prev, rhs)
            # self.bc.apply(rhs)

            solver.solve(u, rhs)
            out.store(u.copy(), t)
            u_prev = u.copy()
        
        return
    
    def solveAdj(
        self, 
        out: hl.TimeDependentVector, 
        x
    ) -> None:

        # Remove any previous data from output vector
        out.zero()

        # Initialise adjoint vector and right-hand side
        p = dl.Vector()
        self.M.init_vector(p, 0)
        rhs = dl.Vector()
        self.M.init_vector(rhs, 0)

        # Initialise previous (next?) adjoint vector (initial condition: p=0)
        p_prev = dl.Vector()
        self.M.init_vector(p_prev, 0)

        # Convert parameter to function
        k = self.vec2func(x[hl.PARAMETER], hl.PARAMETER)

        # Assemble stiffness matrix
        K = ufl.exp(k) * ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        K = dl.assemble(K)

        # Compute gradient of misfit with respect to state
        grad_state = hl.TimeDependentVector(self.ts)
        grad_state.initialize(M=self.M, dim=0)
        self.misfit.grad(hl.STATE, x, grad_state)

        # Define LHS of variational form
        A = self.M + self.dt * K
        # self.bc_p.apply(A)
        solver = dl.LUSolver(A)

        # Snapshot of gradient of state
        grad_state_snap = dl.Vector()
        self.M.init_vector(grad_state_snap, 0)

        for t in self.ts[::-1]:

            grad_state.retrieve(grad_state_snap, t)
            self.M.mult(p_prev - grad_state_snap, rhs)
            # self.bc_p.apply(rhs)

            solver.solve(p, rhs)
            out.store(p.copy(), t)
            p_prev = p.copy()

        return
            
    def evalGradientParameter(self,x, mg, misfit_only=False):
        
        self.prior.init_vector(mg, 1)

        if misfit_only == False:
            dm = x[hl.PARAMETER] - self.prior.mean
            self.prior.R.mult(dm, mg)
        else:
            mg.zero()
        
        p0 = dl.Vector()
        self.M.init_vector(p0,0)
        x[hl.ADJOINT].retrieve(p0, self.ts[1])
        
        # mg.axpy(-1., self.Mt_stab*p0)
        
        g = dl.Vector()
        self.M.init_vector(g,1)
        
        self.prior.Msolver.solve(g,mg)
        
        grad_norm = g.inner(mg)
        
        return grad_norm
    
    def setPointForHessianEvaluations(
        self, 
        x, 
        gauss_newton_approx: bool = True
    ) -> None:
        """Specifies the point x = [u,a,p] at which the (Gauss-Newton) 
        Hessian needs to be evaluated.
        
        Nothing to do since the problem is linear (TODO: why?).
        """
        self.gauss_newton_approx = gauss_newton_approx
        self.x = x
        return

    def solveFwdIncremental(
        self, 
        sol: hl.TimeDependentVector, 
        rhs: hl.TimeDependentVector
    ) -> None:
        """Solves the incremental forward problem (with RHS) and saves 
        the result into sol.
        """

        # Remove previous data in solution vector
        sol.zero()

        # Initialise state vector and right-hand side
        u_inc = dl.Vector()
        self.M.init_vector(u_inc, 0)
        rhs_k = dl.Vector()
        self.M.init_vector(rhs_k, 0)
        rhs_snap = dl.Vector()
        self.M.init_vector(rhs_snap, 0)

        # Initialise previous state
        u_prev = dl.Vector()
        self.M.init_vector(u_prev, 0)

        # Convert parameter to function
        k = self.vec2func(self.x[hl.PARAMETER], hl.PARAMETER)

        # Assemble stiffness matrix
        K = dl.assemble(
            ufl.exp(k) 
                * ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) 
                * ufl.dx
        )
                
        # Define LHS of variational form
        A = self.M + self.dt * K
        # self.bc.apply(A)
        solver = dl.LUSolver(A)
        
        for t in self.ts[1:]:
            
            # Form right-hand side
            rhs.retrieve(rhs_snap, t)
            self.M.mult(u_prev, rhs_k)
            rhs_k.axpy(1.0, rhs_snap)
            
            solver.solve(u_inc, rhs_k)
            sol.store(u_inc.copy(), t)
            u_prev = u_inc.copy()

        return
        
    def solveAdjIncremental(
        self, 
        sol: hl.TimeDependentVector, 
        rhs: hl.TimeDependentVector
    ) -> None:
        """Solves the incremental adjoint problem.
        """

        # Remove any previous data from output vector
        sol.zero()

        # Initialise adjoint vector
        p_inc = dl.Vector()
        self.M.init_vector(p_inc, 0)

        # Initialise RHS for each step
        rhs_k = dl.Vector()
        self.M.init_vector(rhs_k, 0)

        rhs_snap = dl.Vector()
        self.M.init_vector(rhs_snap, 0)

        # Initialise previous (next?) adjoint vector (initial condition: p=0)
        p_prev = dl.Vector()
        self.M.init_vector(p_prev, 0)

        # Convert parameter to function
        k = self.vec2func(self.x[hl.PARAMETER], hl.PARAMETER)

        # Assemble stiffness matrix
        K = dl.assemble(
            ufl.exp(k) 
                * ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) 
                * ufl.dx
        )

        # Define LHS of variational form
        A = self.M + self.dt * K
        # self.bc_p.apply(A)
        solver = dl.LUSolver(A)

        for t in self.ts[::-1]:

            # Form right-hand side
            rhs.retrieve(rhs_snap, t)
            self.M.mult(p_prev, rhs_k)
            rhs_k.axpy(1.0, rhs_snap)

            solver.solve(p_inc, rhs_k)
            sol.store(p_inc.copy(), t)
            p_prev = p_inc.copy()

        return
    
    def applyC(
        self, 
        dm: dl.Vector, 
        out: hl.TimeDependentVector
    ) -> None:
        """
        vector to multiply c with.
        out=where result of matrix-vector multiplication will be stored.
        """

        out.zero()

        u = self.x[hl.STATE]
        k = self.vec2func(self.x[hl.PARAMETER], hl.PARAMETER)
        
        u_k = dl.Vector()
        self.M.init_vector(u_k, 0)

        out_k = dl.Vector()
        self.M.init_vector(out_k, 0)

        for t in self.ts[1:]:

            u.retrieve(u_k, t)
            u_k_func = self.vec2func(u_k.copy(), hl.STATE)

            # Build A matrix
            N = dl.assemble(
                self.dt * self.v * ufl.exp(k) 
                    * ufl.inner(ufl.grad(self.u), ufl.grad(u_k_func)) 
                    * ufl.dx
            )
            # self.bc.apply(N)  # TODO: check this.

            N.mult(dm, out_k)
            out.store(out_k, t)

        return
    
    def applyCt(
        self, 
        dp: hl.TimeDependentVector, 
        out: dl.Vector
    ):
        
        out.zero()

        u = self.x[hl.STATE]
        k = self.vec2func(self.x[hl.PARAMETER], hl.PARAMETER)
        
        u_snap = dl.Vector()
        self.M.init_vector(u_snap, 0)

        dp_k = dl.Vector()
        self.M.init_vector(dp_k, 0)

        out_k = dl.Vector()
        self.M.init_vector(out_k, 0)

        for t in self.ts[1:]:  # TODO: check whether the 1: is needed

            # Build A matrix
            u.retrieve(u_snap, t)
            u_k = self.vec2func(u_snap.copy(), hl.STATE)
            
            N = self.dt * dl.assemble(
                self.u * ufl.exp(k) 
                    * ufl.inner(ufl.grad(self.v), ufl.grad(u_k)) 
                    * ufl.dx
            )
            # self.bc.apply(N)  # TODO: check this.

            dp.retrieve(dp_k, t)
            N.mult(dp_k, out_k)
            out.axpy(1.0, out_k)

        return
    
    def applyWuu(self, du, out):
        out.zero()
        self.misfit.apply_ij(hl.STATE, hl.STATE, du, out)
        return

    def applyWum(self, dm, out):
        out.zero()
        return
    
    def applyWmu(self, du, out):
        out.zero()
        return
    
    def applyR(self, dm, out):
        self.prior.R.mult(dm, out)
        return
    
    def applyWmm(self, dm, out):
        out.zero()
        return
        
    def exportState(self, x, filename, varname):
        raise NotImplementedError()