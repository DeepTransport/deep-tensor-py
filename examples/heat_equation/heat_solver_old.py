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
import ufl

from hippylib import *
import hippylib as hl


class SpaceTimePointwiseStateObservation(Misfit):
    def __init__(self, Vh,
                 observation_times,
                 targets,
                 d = None,
                 noise_variance=None):
        
        self.Vh = Vh
        self.observation_times = observation_times
        
        self.B = assemblePointwiseObservation(self.Vh, targets)
        self.ntargets = targets
        
        if d is None:
            self.d = TimeDependentVector(observation_times)
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
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            obs.store(self.Bu_snapshot, t)
            
    def cost(self, x):
        c = 0
        for t in self.observation_times:
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            self.d.retrieve(self.d_snapshot, t)
            self.Bu_snapshot.axpy(-1., self.d_snapshot)
            c += self.Bu_snapshot.inner(self.Bu_snapshot)
            
        return c/(2.*self.noise_variance)
    
    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            for t in self.observation_times:
                x[STATE].retrieve(self.u_snapshot, t)
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
        if i == STATE and j == STATE:
            for t in self.observation_times:
                direction.retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                out.store(self.u_snapshot, t)
        else:
            pass    


class HeatEquation(object):    
    
    def __init__(self, mesh, Vh, prior, misfit, simulation_times):
        
        self.mesh = mesh
        self.Vh = Vh
        self.prior = prior
        self.misfit = misfit
        
        # Assume constant timestepping (TODO: tidy up)
        self.simulation_times = simulation_times
        dt = simulation_times[1] - simulation_times[0]

        u = dl.TrialFunction(Vh[STATE])
        v = dl.TestFunction(Vh[STATE])
        
        kappa = dl.Constant(0.001)  # diffusivity (TODO: make this a function eventually)
        
        # TODO: derive weak form
        # TODO: figure out whether the transpose stuff in here is necessary
        self.M = dl.assemble(ufl.inner(u, v) * ufl.dx)
        self.M_stab = dl.assemble(ufl.inner(u, v) * ufl.dx)
        self.Mt_stab = dl.assemble(ufl.inner(v, u) * ufl.dx)

        Nvarf = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
        Ntvarf = ufl.inner(kappa * ufl.grad(v), ufl.grad(u)) * ufl.dx
        self.N  = dl.assemble(Nvarf)
        self.Nt = dl.assemble(Ntvarf)
        self.L = self.M + dt * self.N
        self.Lt = self.M + dt * self.Nt
        
        self.solver = PETScLUSolver(self.mesh.mpi_comm())
        self.solver.set_operator(dl.as_backend_type(self.L))
        
        self.solvert = PETScLUSolver(self.mesh.mpi_comm()) 
        self.solvert.set_operator(dl.as_backend_type(self.Lt))
                        
        # Part of model public API
        self.gauss_newton_approx = False
        return
                    
    def generate_vector(self, component="ALL"):

        if component == "ALL":
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.M, 0)
            m = dl.Vector()
            self.prior.init_vector(m,0)
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.M, 0)
            return [u, m, p]
        
        elif component == STATE:
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.M, 0)
            return u
        
        elif component == PARAMETER:
            m = dl.Vector()
            self.prior.init_vector(m,0)
            return m
        
        elif component == ADJOINT:
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.M, 0)
            return p
        
        else:
            raise Exception("Unknown component.")
    
    def init_parameter(self, m):
        self.prior.init_vector(m, 0)
          
    def cost(self, x):
        Rdx = dl.Vector()
        self.prior.init_vector(Rdx,0)
        dx = x[PARAMETER] - self.prior.mean
        self.prior.R.mult(dx, Rdx)
        reg = 0.5 * Rdx.inner(dx)
        misfit = self.misfit.cost(x)
        return [reg+misfit, reg, misfit]
    
    def solveFwd(
        self, 
        out: hl.TimeDependentVector, 
        x: list
    ) -> None:
        """Solves the forward problem.
        
        Parameters
        ----------
        out:
            A set of state vectors at each snapshot. Each vector is 
            stored in out.data.
        x:
            State / parameter / adjoint vectors.
        
        """
    
        # Remove old data in state vector
        out.zero()

        uold = x[PARAMETER]
        u = dl.Vector()
        rhs = dl.Vector()
        self.M.init_vector(rhs, 0)
        self.M.init_vector(u, 0)
        for t in self.simulation_times[1::]:
            self.M_stab.mult(uold, rhs)  # computes M_stab @ rhs and saves in u_old
            self.solver.solve(u, rhs)
            out.store(u,t)
            uold = u
        
        return
    
    def solveAdj(self, out, x):
        
        grad_state = TimeDependentVector(self.simulation_times)
        grad_state.initialize(self.M, 0)
        self.misfit.grad(STATE, x, grad_state)
        
        out.zero()
        
        pold = dl.Vector()
        self.M.init_vector(pold,0)
            
        p = dl.Vector()
        self.M.init_vector(p,0)
        
        rhs = dl.Vector()
        self.M.init_vector(rhs,0)
        
        grad_state_snap = dl.Vector()
        self.M.init_vector(grad_state_snap,0)

          
        for t in self.simulation_times[::-1]:
            self.Mt_stab.mult(pold,rhs)
            grad_state.retrieve(grad_state_snap, t)
            rhs.axpy(-1., grad_state_snap)
            self.solvert.solve(p, rhs)
            pold = p
            out.store(p, t)            
            
    def evalGradientParameter(self,x, mg, misfit_only=False):
        self.prior.init_vector(mg,1)
        if misfit_only == False:
            dm = x[PARAMETER] - self.prior.mean
            self.prior.R.mult(dm, mg)
        else:
            mg.zero()
        
        p0 = dl.Vector()
        self.M.init_vector(p0,0)
        x[ADJOINT].retrieve(p0, self.simulation_times[1])
        
        mg.axpy(-1., self.Mt_stab*p0)
        
        g = dl.Vector()
        self.M.init_vector(g,1)
        
        self.prior.Msolver.solve(g,mg)
        
        grad_norm = g.inner(mg)
        
        return grad_norm
        
    
    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        """
        Specify the point x = [u,a,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated.
        
        Nothing to do since the problem is linear
        """
        self.gauss_newton_approx = gauss_newton_approx
        return

        
    def solveFwdIncremental(self, sol, rhs):
        sol.zero()
        uold = dl.Vector()
        u = dl.Vector()
        Muold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(uold, 0)
        self.M.init_vector(u, 0)
        self.M.init_vector(Muold, 0)
        self.M.init_vector(myrhs, 0)

        for t in self.simulation_times[1::]:
            self.M_stab.mult(uold, Muold)
            rhs.retrieve(myrhs, t)
            myrhs.axpy(1., Muold)
            self.solver.solve(u, myrhs)
            sol.store(u,t)
            uold = u


        
    def solveAdjIncremental(self, sol, rhs):
        sol.zero()
        pold = dl.Vector()
        p = dl.Vector()
        Mpold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(pold, 0)
        self.M.init_vector(p, 0)
        self.M.init_vector(Mpold, 0)
        self.M.init_vector(myrhs, 0)

        for t in self.simulation_times[::-1]:
            self.Mt_stab.mult(pold,Mpold)
            rhs.retrieve(myrhs, t)
            Mpold.axpy(1., myrhs)
            self.solvert.solve(p, Mpold)
            pold = p
            sol.store(p, t)  
            
    
    def applyC(self, dm, out):
        out.zero()
        myout = dl.Vector()
        self.M.init_vector(myout, 0)
        self.M_stab.mult(dm,myout)
        myout *= -1.
        t = self.simulation_times[1]
        out.store(myout,t)
        
        myout.zero()
        for t in self.simulation_times[2:]:
            out.store(myout,t)
    
    def applyCt(self, dp, out):
        t = self.simulation_times[1]
        dp0 = dl.Vector()
        self.M.init_vector(dp0,0)
        dp.retrieve(dp0, t)
        dp0 *= -1.
        self.Mt_stab.mult(dp0, out)

    
    def applyWuu(self, du, out):
        out.zero()
        self.misfit.apply_ij(STATE, STATE, du, out)

    
    def applyWum(self, dm, out):
        out.zero()

    
    def applyWmu(self, du, out):
        out.zero()
    
    def applyR(self, dm, out):
        self.prior.R.mult(dm,out)
    
    def applyWmm(self, dm, out):
        out.zero()
        
    def exportState(self, x, filename, varname):
        out_file = dl.XDMFFile(self.Vh[STATE].mesh().mpi_comm(), filename)
        out_file.parameters["functions_share_mesh"] = True
        out_file.parameters["rewrite_function_mesh"] = False
        ufunc = dl.Function(self.Vh[STATE], name=varname)
        t = self.simulation_times[0]
        out_file.write(vector2Function(x[PARAMETER], self.Vh[STATE], name=varname),t)
        for t in self.simulation_times[1:]:
            x[STATE].retrieve(ufunc.vector(), t)
            out_file.write(ufunc, t)