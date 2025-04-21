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
        
        c = 0.0

        for t in self.observation_times:
            x[hl.STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            self.d.retrieve(self.d_snapshot, t)
            self.Bu_snapshot.axpy(-1., self.d_snapshot)
            # print(self.Bu_snapshot[:])
            c += self.Bu_snapshot.inner(self.Bu_snapshot)
            
        return 0.5 * c / self.noise_variance
    
    def grad(self, i, x, out):
        """TODO: check this."""
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
        self.nt = len(ts)
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
            c=5.0*np.pi*1e-2,
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
        """Returns a single vector sampled from the prior.
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
        self.prior.init_vector(Rdx, 0)
        dx = x[hl.PARAMETER] - self.prior.mean
        self.prior.R.mult(dx, Rdx)
        reg = 0.5 * Rdx.inner(dx)
        misfit = self.misfit.cost(x)
        return [reg + misfit, reg, misfit]
    
    def Rsolver(self):        
        return self.prior.Rsolver
    
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
        
        for t in self.ts:

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
            self.M.mult(p_prev, rhs)
            rhs.axpy(-1.0, grad_state_snap)
            # self.bc_p.apply(rhs)

            solver.solve(p, rhs)
            out.store(p.copy(), t)
            p_prev = p.copy()

        return
            
    def evalGradientParameter(
        self, 
        x, 
        mg, 
        misfit_only: bool = False
    ):
        
        self.x = x

        if misfit_only:
            print("misfit only")

        self.prior.init_vector(mg, dim=1)

        d_param = x[hl.PARAMETER] - self.prior.mean
        self.prior.R.mult(d_param, mg)
        
        p0 = self.generate_vector(hl.PARAMETER)
        p0.zero()

        u = self.x[hl.STATE]
        k = self.vec2func(self.x[hl.PARAMETER], hl.PARAMETER)
        p = self.x[hl.ADJOINT]
        
        u_snap = dl.Vector()
        self.M.init_vector(u_snap, 0)

        p_snap = dl.Vector()
        self.M.init_vector(p_snap, 0)

        for t in self.ts:

            # Build A matrix
            u.retrieve(u_snap, t)
            p.retrieve(p_snap, t)
            u_k = self.vec2func(u_snap.copy(), hl.STATE)
            p_k = self.vec2func(p_snap.copy(), hl.ADJOINT)
            
            grad_k = self.dt * dl.assemble(
                self.v * ufl.exp(k) 
                    * ufl.inner(ufl.grad(p_k), ufl.grad(u_k)) 
                    * ufl.dx
            )

            p0.axpy(1.0, grad_k)

        mg.axpy(1.0, p0)
        grad_norm = mg.inner(mg)
        return grad_norm
    
    def setPointForHessianEvaluations(
        self, 
        x, 
        gauss_newton_approx: bool = True
    ) -> None:
        """Specifies the point x = [u,a,p] at which the Gauss-Newton)
        Hessian needs to be evaluated.
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
        
        for t in self.ts:
            
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
            self.M.mult(p_prev, rhs_k)
            rhs.retrieve(rhs_snap, t)
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

        for t in self.ts:

            u.retrieve(u_k, t)
            u_k_func = self.vec2func(u_k.copy(), hl.STATE)

            # Build A matrix
            N = dl.assemble(
                self.dt * self.u * ufl.exp(k) 
                    * ufl.inner(ufl.grad(self.v), ufl.grad(u_k_func)) 
                    * ufl.dx
            )

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

        for t in self.ts:  # TODO: check (should this be ts[:-1]?)

            # Build A matrix
            u.retrieve(u_snap, t)
            u_k = self.vec2func(u_snap.copy(), hl.STATE)
            
            N = self.dt * dl.assemble(
                self.v * ufl.exp(k) 
                    * ufl.inner(ufl.grad(self.u), ufl.grad(u_k)) 
                    * ufl.dx
            )

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
    
    def _build_MK(self, k: dl.Vector):
        """Builds mass matrices and stiffness vectors."""

        k = self.vec2func(k, hl.PARAMETER)
    
        M = dl.assemble(self.u * self.v * ufl.dx)
        K = dl.assemble(ufl.exp(k) * ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx)
        
        return M.array(), K.array()
        
    def _build_A(self, M: np.ndarray, K: np.ndarray, n_u: int):
        """Builds big A matrix."""

        A = np.zeros((n_u*self.nt, n_u*self.nt))
        
        for i in range(self.nt):
            i0 = n_u*i 
            i1 = n_u*(i+1)
            A[i0:i1, :][:, i0:i1] = M + self.dt * K
        
        for i in range(self.nt-1):
            i0 = n_u*(i+1)
            i1 = n_u*(i+2)
            j0 = n_u*i
            j1 = n_u*(i+1)
            A[i0:i1, :][:, j0:j1] = -M

        return A

    def check_forward(self, x, ):

        u, k, _ = x
        n_u = len(u.data[0][:])

        M, K = self._build_MK(k)
        A = self._build_A(M, K, n_u)

        M = self.M.array()
        f = self.f[:]
        u0 = self.u0[:]
        b = np.hstack([self.dt * M @ f + M @ u0, *[self.dt * M @ f for _ in range(self.nt-1)]])

        uu = np.linalg.solve(A, b)

        self.solveFwd(u, x)
        uu_true = np.hstack([u.data[i] for i in range(self.nt)])

        print(np.max(np.abs(uu-uu_true)))

        return
    
    def check_adjoint(self, x):

        u, k, p = x
        p = self.generate_vector(hl.ADJOINT)
        n_u = len(u.data[0][:])

        M, K = self._build_MK(k)
        A = self._build_A(M, K, n_u)

        M = self.M.array()
        f = self.f[:]
        u0 = self.u0[:]
        b = np.hstack([self.dt * M @ f + M @ u0, *[self.dt * M @ f for _ in range(self.nt-1)]])

        B = self.misfit.B.array()
        noise_var = self.misfit.noise_variance

        rhs = np.zeros((self.nt, n_u))
        for i, t in enumerate(self.ts):
            if np.min(np.abs(t - self.misfit.observation_times)) < 1e-8:
                u_i = u.data[i][:]
                d_i = dl.Vector()
                self.misfit.B.init_vector(d_i, 0)
                self.misfit.d.retrieve(d_i, t)
                rhs[i] = (1 / noise_var) * B.T @ (B @ u_i - d_i)

        #rhs = np.hstack([(1 / noise_var) * B.T @ (B @ ...) for i in range(self.nt)])
        rhs = rhs.flatten()

        pp = np.linalg.solve(A.T, -rhs)
        self.solveAdj(p, x)
        pp_true = np.hstack([p.data[i] for i in range(self.nt)])

        print(np.max(np.abs(pp-pp_true)))

        return

    def check_dAuda_fd(self, k: dl.Vector, u: hl.TimeDependentVector):
        
        def _build_dAuda_true(k: dl.Vector, u: hl.TimeDependentVector):
            """Builds true dAdu."""

            dAuda_true = np.zeros((n_u*(self.nt-1), n_k))
            k_func = self.vec2func(k.copy(), hl.PARAMETER)

            u_k = dl.Vector()
            self.M.init_vector(u_k, 0)
            
            for i, t in enumerate(self.ts[1:]):

                u.retrieve(u_k, t)
                u_i_func = self.vec2func(u_k.copy(), hl.STATE)

                # Build A matrix
                N = dl.assemble(
                    self.dt * self.u * ufl.exp(k_func) 
                        * ufl.inner(ufl.grad(self.v), ufl.grad(u_i_func)) 
                        * ufl.dx
                )

                i0 = n_u*i 
                i1 = n_u*(i+1)
                dAuda_true[i0:i1, :] = N.array()
            
            return dAuda_true

        n_u = len(u.data[0])
        n_k = len(k)

        uu = np.hstack([u.data[i+1][:] for i in range(self.nt-1)])

        dAuda = np.zeros((n_u*(self.nt-1), n_k))
        dx = 1e-8

        for i in range(n_k):
            k_0 = k.copy()
            k_1 = k.copy()
            k_0[i] -= dx
            k_1[i] += dx

            M_0, K_0 = self._build_MK(k_0)
            M_1, K_1 = self._build_MK(k_1)
            A_0 = self._build_A(M_0, K_0, n_u)
            A_1 = self._build_A(M_1, K_1, n_u)

            dAuda[:, i] = ((A_1 - A_0) @ uu) / (2.0*dx)

        dAuda_true = _build_dAuda_true(k, u)

        # Check

        # x_rand = dl.Vector()
        # self.M.init_vector(x_rand, 0)
        # xis = np.random.rand(n_k)
        # x_rand.set_local(xis)

        # u_test = self.generate_vector(hl.STATE)
        # self.x = [u, k, None]
        # self.applyC(x_rand, u_test)
        # uu_test = np.hstack([u_test.data[i+1][:] for i in range(self.nt-1)])

        # # Application of C, manual derivative, manual FD derivative
        # print(uu_test)
        # print(dAuda @ x_rand[:])
        # print(dAuda_true @ x_rand[:])

        # Check transpose

        # u_rand = self.generate_vector(hl.STATE)
        # for i in range(self.nt-1):
        #     xis = np.random.rand(n_u)
        #     u_rand.data[i+1].set_local(xis)

        # uu_rand = np.hstack([u_rand.data[i+1][:] for i in range(self.nt-1)])
        # k_test = self.generate_vector(hl.PARAMETER)
        # self.applyCt(u_rand, k_test)

        # print(k_test[:][:10])
        # print((dAuda.T @ uu_rand)[:10])
        # print((dAuda_true.T @ uu_rand)[:10])

        return

    def lagrangian(self, x):

        u, k, p = x

        uu = np.hstack([u.data[i+1] for i in range(self.nt-1)])
        kk = k[:]
        pp = np.hstack([p.data[i] for i in range(self.nt-1)])

        n_k = len(k[:])
        M, K = self._build_MK(k)
        A = self._build_A(M, K, n_k)

        ## Form b
        M = self.M.array()
        f = self.f[:]
        u0 = self.u0[:]

        b = np.hstack([self.dt * M @ f + M @ u0, *[self.dt * M @ f for _ in range(self.nt-2)]])

        L = pp.T @ (A @ uu - b) + self.cost(x)[0]
        return L

    def check_gradient_fd(self, k: dl.Vector):
        """Conducts a finite difference check of the gradient of the 
        Lagrangian with respect to the parameter at a given parameter 
        vector.
        """

        u = self.generate_vector(hl.STATE)
        p = self.generate_vector(hl.ADJOINT)
        x = [u, k, p]
        self.solveFwd(u, x)
        self.solveAdj(p, x)

        n_k = len(k[:])

        dx = 1e-6
        
        dldk_fd = np.zeros((n_k, ))

        for i in range(n_k):

            k_0 = k.copy()
            k_1 = k.copy()

            k_0[i] -= dx
            k_1[i] += dx

            l_0 = self.lagrangian([u, k_0, p])
            l_1 = self.lagrangian([u, k_1, p])

            dldk_fd[i] = (l_1-l_0) / (2*dx)

        dldk = dl.Vector()
        self.M.init_vector(dldk, 0)
        self.evalGradientParameter(x, dldk)
        
        return

    def exportState(self, x, filename, varname):
        raise NotImplementedError()