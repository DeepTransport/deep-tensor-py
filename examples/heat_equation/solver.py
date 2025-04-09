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
            pass    


class HeatSolver(object):
    
    def __init__(
        self, 
        mesh, 
        Vh, 
        prior: hl.modeling._Prior, 
        misfit, 
        ts: np.ndarray
    ):
        
        self.mesh = mesh
        self.Vh = Vh
        self.prior = prior
        self.misfit = misfit

        self.dt = ts[1] - ts[0]  # NOTE: assumes equispaced times
        self.ts = ts

        self.u = dl.TrialFunction(Vh[hl.STATE])
        self.v = dl.TestFunction(Vh[hl.STATE])

        self.u0 = dl.Constant(0.0)
        
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

        # Assemble mass matrix
        self.M = dl.assemble(self.u * self.v * ufl.dx)
        
        self.bc = dl.DirichletBC(
            self.Vh[hl.STATE], 
            dl.Constant(0.0), 
            self.dirichlet_boundary
        )

        # Part of model public API (??)
        self.gauss_newton_approx = False
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

    def sample_prior(self):
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
        # Convert to function
        k = dl.Function(self.Vh[hl.PARAMETER])
        k.vector().set_local(ks.get_local()[:])
        k.vector().apply("insert")
        return k

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
            self.prior.init_vector(M=m, dim=0)
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
        
        u = dl.Function(self.Vh[hl.STATE])
        # k = dl.Function(self.Vh[hl.PARAMETER])
        # k.vector().set_local(x[hl.PARAMETER].get_local()[:])
        # k.vector().apply("insert")

        # Assemble stiffness matrix
        K = ufl.exp(x[hl.PARAMETER]) * ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        K = dl.assemble(K)
                
        # Define LHS of variational form
        A = self.M + self.dt * K
        # A = dl.assemble(a)

        # Define previous state
        u_prev = dl.interpolate(self.u0, self.Vh[hl.STATE])
        
        for t in self.ts[1:]:
            
            # Assemble RHS vector
            L = (u_prev + self.dt * self.f) * self.v * ufl.dx
            b = dl.assemble(L)

            self.bc.apply(A, b)
            dl.solve(A, u.vector(), b)
            
            out.store(u.vector().copy(), t)
            u_prev.assign(u)

        # May need to do something similar to this to update 
        # parameter in forcing term
        # self.u0.t = t  # Update t parameter in initial condition

        # May also need to apply boundary conditions in here
        # (currently no need as they are no-flow)
        
        return
    
    def solveAdj(self, out, x):
        raise NotImplementedError()
            
    def evalGradientParameter(self,x, mg, misfit_only=False):
        raise NotImplementedError()
    
    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        raise NotImplementedError()

    def solveFwdIncremental(self, sol, rhs):
        raise NotImplementedError()
        
    def solveAdjIncremental(self, sol, rhs):
        raise NotImplementedError()
    
    def applyC(self, dm, out):
        raise NotImplementedError()
    
    def applyCt(self, dp, out):
        raise NotImplementedError()
    
    def applyWuu(self, du, out):
        raise NotImplementedError()
    
    def applyWum(self, dm, out):
        raise NotImplementedError()
    
    def applyWmu(self, du, out):
        raise NotImplementedError()
    
    def applyR(self, dm, out):
        raise NotImplementedError()
    
    def applyWmm(self, dm, out):
        raise NotImplementedError()
        
    def exportState(self, x, filename, varname):
        raise NotImplementedError()