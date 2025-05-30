'''
 Code used to solve the PAI forward problem, and construct jacobian via adjoints 
 
  solves the coupled PDEs: p(t_i,x_j) = B G(m)
  
  G(m) = mu_a(m)*phi(m), phi(m) obtained via solving optic PDE:
  mu_a(m) = exp(m), kappa(mu_a) = 1/(2*(mu_a+0.2*mu_s)), mu_s fixed constant 
  mu_a phi - \div(kappa(mu_a) \grad(phi)) = 0 in Omega
  (1/pi)*phi - 0.5*kappa(mu_a)*\grad(phi)*n = s on dOmega
  p0 = mu_a*phi

  B: passed in matrix, takes p_0, outputs p(t_i,x_j) at 5 observation times, 50 x_js via solution of acoustic wave equation: 
  p_tt - c^2 \Delta p = 0 in Omega x [0,T]
  p(t = 0) = p0 in Omega
  p_t(t = 0) = 0 in Omega
  Boundary conditions are chosen to approximate an absorbing boundary
'''

import matplotlib.pyplot as plt
import dolfin as dl
# from mshr import * 
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
import numpy as np
import sys
from hippylib import *


class AcousticMisfit(Misfit):
	def __init__(self, Vh, obsOp, d = None, noise_variance=None):
		
		self.Vh = Vh
		self.B = obsOp # numpy array 

		if d is None:
			self.d = np.zeros((self.B.shape[0],1))
		else:
			self.d = d
			
		self.noise_variance = noise_variance
		
	def observe(self,p0):
		return self.B @ (p0.vector().get_local())
		
	def cost(self,p0):
		if self.noise_variance is None: 
			raise ValueError("Noise Variance must be specified")
		elif self.noise_variance == 0:
			raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
		Bu = self.B @ p0.vector().get_local() - self.d
		return (.5/self.noise_variance)*np.inner(Bu,Bu)
        
	def grad(self, p0):
		if self.noise_variance is None:
			raise ValueError("Noise Variance must be specified")
		elif self.noise_variance == 0:
			raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
		Bu = (1./self.noise_variance)*(self.B @ p0.vector().get_local() - self.d)
		BTBu = self.B.T @ Bu
		
		return BTBu 
			
class DiffusionApproximation:
	# Class containing Fwd PDE solve for the diffusion approximation to the RTE
	def __init__(self,mesh,Vh,prior,misfit,mu_s_nominal):	
		# Mesh and finite element space information
		self.mesh = mesh
		self.Vh = Vh # assumes Vh is the same for state, param, and adjoint 
		
		# prior information
		self.prior = prior
		self.misfit = misfit
		
		# Fixed params 
		self.mu_s = dl.Constant(mu_s_nominal)
		
		# light source, only on top and bottom boundary 
		boundary_expression = dl.Expression("((x[1] < DOLFIN_EPS || abs(x[1] - 3.0) < DOLFIN_EPS) ? 3.0 * exp(-0.5 * pow(x[0] - xc, 2) / pow(sigma, 2)) : 0.0)", degree=2, xc=2.5, sigma=5.0)
		self.s = dl.interpolate(boundary_expression, self.Vh)
		
		u_trial = dl.TrialFunction(self.Vh)
		u_test = dl.TestFunction(self.Vh)
		
		# mass matrix + LU decomposition
		self.M = dl.assemble(dl.inner(u_trial,u_test)*dl.dx, form_compiler_parameters={"quadrature_degree": 10})
		self.matM = dl.as_backend_type(self.M).mat()
		self.M_csc = spsp.csc_matrix(self.matM.getValuesCSR()[::-1], shape=self.matM.size)
		self.M_LU = spla.splu(self.M_csc)	
			
	def solveFwd(self,x):
		'''
		Solve optical PDE 
		Input: x: m : log(mu_a)  
		Output: [phi,p_0,mu_a]: three functions:
		                               phi = light fluence
		                               p_0 = mu_a*light_fluence
		                               mu_a = absorption coeffient = exp(m) 
		'''
		mu_a = dl.Function(self.Vh)
		dl.project(dl.exp(x),self.Vh,function=mu_a)
		kappa = 1.0 / (2.0 * (mu_a + 0.2 * self.mu_s))
		
		u_trial = dl.TrialFunction(self.Vh)
		u_test = dl.TestFunction(self.Vh)
			
		A = dl.inner(kappa*dl.grad(u_trial),dl.grad(u_test))*dl.dx+dl.inner(mu_a*u_trial,u_test)*dl.dx+(2.0/np.pi)*dl.inner(u_trial,u_test)*dl.ds
		
		rhs = 2.0*dl.inner(self.s,u_test)*dl.ds
		
		phi = dl.Function(self.Vh)
		dl.solve(A == rhs, phi)
		
		p_0 = dl.Function(self.Vh)
		dl.project(mu_a*phi, self.Vh, function=p_0)
		
		return [phi,p_0,mu_a]
		
	def computeJacobian(self,x):
		''' 
		Compute the Jacobian of PTO map at (mu_a = x[0], phi = x[1])
		'''
		# mu_a,phi,kappa,dmu_a/dm, dkappa/dm
		mu_a = x[0]
		phi = x[1]
		kappa = 1.0 / (2.0 * (mu_a + 0.2 * self.mu_s))
		dmu_a_dm = mu_a
		dkappa_dmu_a = -mu_a/(2.0 * (mu_a + 0.2 * self.mu_s)**2)
		
		u_trial = dl.TrialFunction(self.Vh)
		u_test = dl.TestFunction(self.Vh)
		
		A = dl.inner(kappa*dl.grad(u_trial),dl.grad(u_test))*dl.dx+dl.inner(mu_a*u_trial,u_test)*dl.dx+(2.0/np.pi)*dl.inner(u_trial,u_test)*dl.ds
		
		# compute decomposition for faster solves 
		matA = dl.as_backend_type(dl.assemble(A)).mat()
		A_csc = spsp.csc_matrix(matA.getValuesCSR()[::-1], shape=matA.size)
		A_LU = spla.splu(A_csc)
		
		nObs = misfit.B.shape[0]
		
		Jac = np.zeros((self.Vh.dim(),nObs))
		G_vec = dl.Function(self.Vh)
		J_col = dl.Function(self.Vh)
		adjoint = dl.Function(self.Vh)
		
		MinvBT = self.M_LU.solve(self.misfit.B.T)
		print(sep,'building jacobian',sep)
		for i in range(0,nObs):
			G_vec.vector().set_local(MinvBT[:,i])
			rhs = dl.assemble(dl.inner(mu_a * G_vec, u_test) * dl.dx)
			adjoint.vector().set_local(A_LU.solve(rhs.get_local()))
			J_col_form = -dl.inner(dkappa_dmu_a*dl.grad(adjoint),dl.grad(phi)) - dl.inner(mu_a*adjoint,phi) + dl.inner(mu_a*G_vec,phi)
			dl.project(J_col_form, self.Vh, function=J_col)
			Jac[:,i] = J_col.vector().get_local()
		Jac_final = Jac.T @ self.M_csc
		return Jac_final
	
	def testsAdjoint(self,x,xhat,X):
	    # tests if <p0_hat,X> = <m_hat,d(m,adjoint,X)> 
	
		mu_a = x[0]
		phi = x[1]
		m_hat = xhat
		kappa = 1.0 / (2.0 * (mu_a + 0.2 * self.mu_s))
		dmu_a_dm = mu_a
		dkappa_dmu_a = -mu_a/(2.0 * (mu_a + 0.2 * self.mu_s)**2)
		
		# lhs 
		u_test = dl.TestFunction(self.Vh)
		u_trial = dl.TrialFunction(self.Vh)
		A = dl.inner(kappa*dl.grad(u_trial),dl.grad(u_test))*dl.dx+dl.inner(mu_a*u_trial,u_test)*dl.dx+(2.0/np.pi)*dl.inner(u_trial,u_test)*dl.ds
		
		# Compute p0_hat using fwd_incremental
		# rhs 
		rhs_incFwd = -dl.inner(mu_a*phi*m_hat,u_test)*dl.dx
		rhs_incFwd = rhs_incFwd - dl.inner(dkappa_dmu_a*m_hat*dl.grad(phi),dl.grad(u_test))*dl.dx
		
		# solve fwd incremental PDE
		fwd_incremental = dl.Function(self.Vh)
		dl.solve(A == rhs_incFwd, fwd_incremental)
		
		# p_hat = G'(m)m_hat
		p_hat = dl.Function(self.Vh)
		dl.project(mu_a*(fwd_incremental+m_hat*phi), self.Vh, function=p_hat)
		
		# Compute adjoint
		# define rhs
		dJ_dphi = mu_a * X
		rhs_adjoint = dl.inner(dJ_dphi,u_test)*dl.dx 
		# solve for adjoint 
		adjoint = dl.Function(self.Vh)
		dl.solve(A == rhs_adjoint, adjoint)
		
		# <m_hat,d(m,adjoint,X)>
		integral_adjoint = dl.assemble((dmu_a_dm * phi * (X - adjoint) - dl.inner(dkappa_dmu_a * dl.grad(phi), dl.grad(adjoint)) )*m_hat * dl.dx)
		# <p0_hat,X> 
		integral_fwdIncremental = dl.assemble(mu_a*(fwd_incremental+m_hat*phi)*X*dl.dx)
		rel_error = abs(integral_adjoint-integral_fwdIncremental)/abs(integral_fwdIncremental)
		
		return rel_error
		
	def testsAdjoint_Full(self,x,xhat,X):
	    # tests if <Bp0_hat,X> = <m_hat,d(m,adjoint*B'*X)> 
	
		mu_a = x[0]
		phi = x[1]
		m_hat = xhat
		kappa = 1.0 / (2.0 * (mu_a + 0.2 * self.mu_s))
		dmu_a_dm = mu_a
		dkappa_dmu_a = -mu_a/(2.0 * (mu_a + 0.2 * self.mu_s)**2)
		
		# lhs 
		u_test = dl.TestFunction(self.Vh)
		u_trial = dl.TrialFunction(self.Vh)
		A = dl.inner(kappa*dl.grad(u_trial),dl.grad(u_test))*dl.dx+dl.inner(mu_a*u_trial,u_test)*dl.dx+(2.0/np.pi)*dl.inner(u_trial,u_test)*dl.ds
		
		# Compute p0_hat using fwd_incremental
		# rhs
		rhs_incFwd = -dl.inner(mu_a*phi*m_hat,u_test)*dl.dx
		rhs_incFwd = rhs_incFwd - dl.inner(dkappa_dmu_a*m_hat*dl.grad(phi),dl.grad(u_test))*dl.dx
		
		# solve fwd incremental pde 
		fwd_incremental = dl.Function(self.Vh)
		dl.solve(A == rhs_incFwd, fwd_incremental)
		
		# p_hat = G'(m)m_hat
		p_hat = dl.Function(self.Vh)
		dl.project(mu_a*(fwd_incremental+m_hat*phi), self.Vh, function=p_hat)
		
		# Compute B*p_hat, apply observation operator 
		Gp_hat = self.misfit.B @ p_hat.vector().get_local()
		
		# Compute adjoint
		# define rhs, G^*X = Minv * G^T X
		GtX = dl.Function(self.Vh)
		GtX.vector().set_local(self.M_LU.solve(self.misfit.B.T @ X))
		rhs_adjoint = dl.inner(mu_a * GtX, u_test)*dl.dx 
		
		# solve for adjoint 
		adjoint = dl.Function(self.Vh)
		dl.solve(A == rhs_adjoint, adjoint)
		
		# <m_hat,d(m,adjoint,G^*X)>_L2
		integral_adjoint = dl.assemble((dmu_a_dm * phi * (GtX - adjoint) - dl.inner(dkappa_dmu_a * dl.grad(phi), dl.grad(adjoint)) )*m_hat * dl.dx)
		# <Bp_hat,X>_R
		integral_fwdIncremental_top = np.dot(X,Gp_hat)
		
		rel_error = abs(integral_adjoint-integral_fwdIncremental_top)/abs(integral_fwdIncremental_top)
		
		return rel_error
		
	def solveFwd_Incremental(self,xhat,x):
		'''
		solve incremental fwd equation 
		Input: xhat - m_hat
			   x - [x[STATE],x[PARAMETER],~], x[STATE]: phi(mu_a) (fluence), x[PARAMETER]: mu_a
		Output: [u_trial,p0_hat_init] - u_trial = phi_hat, p0_hat_init = initial pressure for incremental acoustic fwd
		'''
		mu_a = x[1]
		phi = x[0]
		kappa = 1.0 / (2.0 * (mu_a + 0.2 * self.mu_s))
		dmu_a_dm = mu_a
		dkappa_dmu_a = -mu_a/(2.0 * (mu_a + 0.2 * self.mu_s)**2)
		
		m_hat = xhat
		
		# Define lhs		
		u_trial = dl.TrialFunction(self.Vh)
		u_test = dl.TestFunction(self.Vh)
		A = dl.inner(kappa*dl.grad(u_trial),dl.grad(u_test))*dl.dx+dl.inner(mu_a*u_trial,u_test)*dl.dx+(2.0/np.pi)*dl.inner(u_trial,u_test)*dl.ds
		
		# Define rhs
		rhs_incFwd = -dl.inner(mu_a*phi*m_hat,u_test)*dl.dx
		rhs_incFwd = rhs_incFwd - dl.inner(dkappa_dmu_a*m_hat*dl.grad(phi),dl.grad(u_test))*dl.dx
		
		# solve pde
		fwd_incremental = dl.Function(self.Vh)
		dl.solve(A == rhs_incFwd, fwd_incremental)
		
		# p0_hat = G'(m)*m_hat
		p0_hat = dl.Function(self.Vh)
		dl.project(mu_a*(fwd_incremental+m_hat*phi), self.Vh, function=p0_hat)
		
		
		return [fwd_incremental,p0_hat]
	
def samplePrior(prior,Vh,pr_samps_file,nsamples):
	
	noise = dl.Vector()
	prior.init_vector(noise,"noise")
	m_samp = dl.Function(Vh) 
	
	m_samp_Fun = dl.Function(Vh,name="sample_prior")
	s_priors = []
	with dl.XDMFFile(mesh.mpi_comm(), pr_samps_file) as fid: 
		fid.parameters["functions_share_mesh"] = True
		fid.parameters["rewrite_function_mesh"] = False
		for i in range(0,nsamples):
			parRandom.normal(1.0,noise)
			prior.sample(noise,m_samp.vector())
			dl.project(dl.exp(m_samp),Vh,function=m_samp_Fun)
			fid.write(m_samp_Fun,i)	
			s_priors.append(m_samp)
	return s_priors
				    
	
def FDgradCheck(prior,misfit,problem,m,m_hat,nObs,observation_times):
	# Finite difference gradient check using forward incremental 
	mtrue = m 
	[fluence,p0,mu_a] = problem.solveFwd(mtrue)
	Gm = misfit.observe(p0)
	
	x = [fluence,mu_a]
	
	[fluence_hat,p_hat] = problem.solveFwd_Incremental(m_hat,x)
	Gpm = misfit.observe(p_hat) # apply observation operator 
	
	n_eps = 8
	eps = np.power(.1, np.arange(0,n_eps))
	eps = eps[::-1]
	
	m_p_mhat = dl.Function(problem.Vh)
	
	err_Gp = np.zeros(n_eps)
	[phi,p0,blah] = problem.solveFwd(mtrue)
	
	Gm = misfit.B @ p0.vector().get_local()
	
	for i in range(0,n_eps):
		eps_i = eps[i]
		
		m_p_mhat.vector().set_local(mtrue.vector().get_local()+eps_i*m_hat.vector().get_local())
		[fluence_m_p_mhat,p0_m_p_mhat,mu_a_m_p_mhat] = problem.solveFwd(m_p_mhat)
		
		Gm_p_mhat = misfit.observe(p0_m_p_mhat) # apply observation operator 
		Gpm_FD = (1/eps_i)*(Gm_p_mhat-Gm)
		
		err_Gp[i] = np.linalg.norm(Gpm-Gpm_FD)/np.linalg.norm(Gpm)
		
	plt.figure()
	plt.loglog(eps, err_Gp, "-ob", eps, eps*(err_Gp[0]/eps[0]),"-.k")
	plt.title("FD Gradient Check using Fwd incremental")
	plt.show()
	
def FDgradCheckJac(prior,misfit,problem,m,m_hat,nObs,observation_times,Jac):
	# Finite difference gradient check using Jacobian
	mtrue = m 
	m_hat_array = m_hat.vector().get_local()

	Gpm = Jac @ m_hat_array 
	
	n_eps = 8
	eps = np.power(.1, np.arange(0,n_eps))
	eps = eps[::-1]
	
	[phi,p0,blah] = problem.solveFwd(mtrue)
	
	Gm = misfit.B @ p0.vector().get_local()
		
	m_p_mhat = dl.Function(problem.Vh)
	
	err_Gp = np.zeros(n_eps)
	
	for i in range(0,n_eps):
		eps_i = eps[i]
		m_p_mhat.vector().set_local(mtrue.vector().get_local()+eps_i*m_hat.vector().get_local())
		[fluence_m_p_mhat,p0_m_p_mhat,mu_a_m_p_mhat] = problem.solveFwd(m_p_mhat)
		
		Gm_p_mhat = misfit.B @ p0_m_p_mhat.vector().get_local() # apply observation operator 
		Gpm_FD = (1/eps_i)*(Gm_p_mhat-Gm)
		
		err_Gp[i] = np.linalg.norm(Gpm-Gpm_FD)/np.linalg.norm(Gpm)

	plt.figure()
	plt.loglog(eps, err_Gp, "-ob", eps, eps*(err_Gp[0]/eps[0]),"-.k")
	plt.title("FD Gradient Check using Jacobian, derivative of PTO map")
	plt.show()

if __name__ == "__main__":
	sep = "\n"+"#"*80+"\n"

	'''
	Set up the mesh
	'''
	mesh = dl.RectangleMesh(dl.Point(0.,0.),dl.Point(5.,3.),80,48) 
	
	rank = dl.MPI.rank(mesh.mpi_comm())
	nproc = dl.MPI.size(mesh.mpi_comm())
	
	''' 
	Set up the function spaces 
	'''
	if rank == 0: 
		print( sep, "set up finite element function spaces", sep)
	
	Vh = dl.FunctionSpace(mesh,"P",1)
	ndofs = Vh.dim()
	
	'''
	Set up time discretization
	'''
	if rank == 0: 
		print("Number of dof for state: {0}".format(ndofs))
		print("Number of dof for one parameter: {0}".format(ndofs))
		print(sep, "setting up simulation times and synthesizing sound speed", sep) 
	
	t_init = 0.
	t_final = 8e-6
	
	dt = 5e-8 
	simulation_times = np.arange(t_init,t_final+0.5*dt,dt)
	if rank == 0: 
		print("final time is: {0}".format(t_final))
		print("timestep size: {0}".format(dt))
		print("number of timesteps: {0}".format(len(simulation_times)))
	
	'''
	Set up prior
	'''
	gamma_prior = 0.5; delta_prior = 0.5;
	pr_mean = dl.interpolate(dl.Constant(-4.0),Vh)
	
	prior = BiLaplacianPrior(Vh, delta_prior, gamma_prior, mean=pr_mean.vector(),robin_bc=True)
	
	# Sample from the prior 
	sample_pr = True
	np.random.seed(42)
	if sample_pr:
		pr_samps_file = 'Plots/PAI/prior_samples.xdmf'
		nsamples = 15
		s_priors = samplePrior(prior,Vh,pr_samps_file,nsamples)
		
	# set random direction for later tests 
	m_hat = dl.Function(Vh)
	m_hat.vector().set_local(s_priors[0].vector().get_local())
	
	'''
	Set up space and time observations
	'''
	dt_obs = 1e-6 
	t1 = 1e-6
	tf = 5e-6
	observation_times = np.arange(t1,tf+.5*dt,dt_obs)
	rel_noise = 0.01
	
	target_start = 0.1
	target_end = 4.9
	dxObs = 0.2
	targets_x = np.arange(target_start,target_end+0.5*dxObs,dxObs)
	targets = np.zeros((50,2))
	
	for i in range(0,targets_x.shape[0]): 
		targets[i,0] = targets_x[i]
		targets[i,1] = 3.0
		
	for i in range(0,targets_x.shape[0]):
		targets[i+targets_x.shape[0],0] = targets_x[i]
		targets[i+targets_x.shape[0],1] = 0.0
		
	nObs = targets.shape[0]
	
	if rank == 0: 
		print('number of spatial observations: {0}'.format(targets.shape[0]))
		print('Total number of observations: {0}', targets.shape[0]*observation_times.shape[0])
		print('observation time step: {0}', dt_obs,sep)
		print('starting ob time: {0}', t1, sep)
		print('ending ob time: {0}', observation_times[-1],sep)
		print('number of observation times: {0}', observation_times.shape[0])
		
	ObsOp = np.load("ObsOperator/PTO_Acoustic_original.npy") # load in observation operator, mapping p_0 (coefficients) to observations 
	misfit = AcousticMisfit(Vh, ObsOp)
	
	'''
	Setup model
	'''
	reduced_scattering = 100.0   # scattering 
	
	problem = DiffusionApproximation(mesh,Vh,prior,misfit,reduced_scattering)
	
	''' 
	Synthesize (or load in) data
	'''
	if rank == 0:
			print(sep,"Synthesizing data",sep) 

	# set ground truth for tests
	m_expr = dl.Expression("((x[0]-cx)*(x[0]-cx))/(a*a) + ((x[1]-cy)*(x[1]-cy))/(b*b) <= 1.0 ? 1.0 : 0.0", degree=2, cx=2.5, cy=1.7, a=1, b=0.5)
	mtrue = dl.interpolate(m_expr, Vh)
	
	fid_truem = dl.XDMFFile('Plots/PAI/true_m.xdmf')
	fid_truem.write(mtrue)
	
	[fluence,p0,mu_a] = problem.solveFwd(mtrue)
	
	d_true = misfit.observe(p0)
	noise_std_dev = 0.1 
	misfit.d = d_true+noise_std_dev*np.random.randn(d_true.shape[0]) 
	misfit.noise_variance = noise_std_dev*noise_std_dev
	
	plots_on = 0
	if plots_on == 1:
		# Visualize pressure data 
		d_2d = d_true.reshape(observation_times.shape[0],targets.shape[0])
		print (sep, 'nobs is: ', nObs) 
		XX,YY = np.meshgrid(np.arange(1,nObs+1),(observation_times)*1e6)
		fig,ax = plt.subplots(1,1)
		color_map = plt.cm.get_cmap('YlGnBu_r')
		reserversed_color_map = color_map.reversed()
		cp = ax.contourf(XX,YY,d_2d,100,cmap=color_map)
		fig.colorbar(cp)
		plt.title('Noisy pressure readings at target points',fontsize=30)
		plt.xlabel('Target number',fontsize=30)
		plt.ylabel('Time (in microseconds)',fontsize=30)
		plt.xticks(fontsize=30)
		plt.yticks(fontsize=30)
		plt.show()
		
	print(sep, 'Adjoint check', sep)
	X = dl.Function(Vh)
	X_samp = samplePrior(prior,Vh,pr_samps_file,1)
	X.vector().set_local((X_samp[0].vector().get_local()))
	rel_errAdj = problem.testsAdjoint([mu_a,fluence],m_hat,X)
	print(f"Relative Error (Adjoint): {rel_errAdj:.4e}")

	X_direction = np.random.rand(misfit.B.shape[0])  # uniform in [0, 1)
	rel_errFullAdj = problem.testsAdjoint_Full([mu_a,fluence],m_hat,X_direction)

	print(f"Relative Error (Adjoint, including obs operator): {rel_errFullAdj:.4e}")
	
	Jac = problem.computeJacobian([mu_a,fluence])
	
	FDgradCheckJac(prior,misfit,problem,mtrue,m_hat,nObs,observation_times,Jac)
	
	sys.exit()
	# Another jacobian test, takes forever
	m_hat_x = dl.Function(Vh)
	direction = np.zeros(misfit.B.shape[1])
	Jac_Fwd = np.zeros(Jac.shape)
	
	for i in range(0,ndofs):
		print(i)
		direction[i] = 1
		m_hat_x.vector().set_local(direction)
		[Fpmhat,test] = problem.solveFwd_Incremental(m_hat_x,[fluence,mu_a])
		d_Fpmhat = misfit.observe(test)
		Jac_Fwd[:,i] = d_Fpmhat
		direction[i] = 0
		
	FDgradCheckJac(prior,misfit,problem,mtrue,m_hat,nObs,observation_times,Jac_Fwd)
	norm_diff = np.linalg.norm(Jac-Jac_Fwd)
	print('rel error, fwd Incremental Jac vs adjoint Jac: {0}'.format(norm_diff/norm_jac))
	
	
	
	
	
			
		

    

   
   
   
   
      
   

