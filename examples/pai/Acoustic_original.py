'''
 Code used to solve the acoustic forward problem 

  p_tt - c^2 \Delta p = 0 in Omega x [0,T]
  p(t = 0) = p0 in Omega
  p_t(t = 0) = 0 in Omega
  Boundary conditions are chosen to approximate an absorbing boundary
  
F(p0): p0 -> p
O: p -> p(ti,xj) pointwise observations 
'''

import matplotlib.pyplot as plt
import dolfin as dl
# from mshr import * 
import scipy.sparse as spsp
from sksparse.cholmod import cholesky
import numpy as np
import sys
import argparse
from hippylib import *
import time


class SpaceTimePointwiseStateObservation(Misfit):
	def __init__(self, Vh, observation_times, targets, d = None, noise_variance=None):
        
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
			pass
            
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

class AcousticWave: 
	# Class containing all necessary PDE solves for solution of Bayesian inverse problem for acoustic wave equation
	# wave equation is solved using Newmark scheme with gamma = 0.5, beta = 0.25
	def __init__(self,mesh,Vh,prior,misfit,simulation_times):
		# Mesh and finite element space information 
		self.mesh = mesh
		self.Vh = Vh # [STATE,PARAM,ADJOINT]
		
		# misfit and prior information 
		self.misfit = misfit
		self.prior = prior 

		# time step information 
		self.simulation_times = simulation_times
		self.sound_speed = 1510.e2
		self.dt = simulation_times[1]-simulation_times[0] # constant timestep is assumed
		self.beta = 0.25
		self.gamma = 0.5 
		## Assemble mass and stiffness matrices and enforce no absorbing boundary conditions
		# trial and test functions
		u = dl.TrialFunction(Vh[STATE])
		v = dl.TestFunction(Vh[STATE])
		
		C_const = dl.interpolate(dl.Expression('c0',c0=1.0/np.sqrt(self.sound_speed),element=self.Vh[STATE].ufl_element()),self.Vh[STATE])
		self.M = dl.assemble( dl.inner((1/self.sound_speed)*u,(1/self.sound_speed)*v)*dl.dx )
		self.S = dl.assemble(dl.inner(C_const*u,C_const*v)*dl.ds) 
		self.K = dl.assemble(dl.inner( dl.grad(u), dl.grad(v) )*dl.dx)
		self.L = (4/(self.dt*self.dt))*self.M+(2/self.dt)*self.S+self.K
		
		# sparse matrices for quick mat vecs in solveFwd
		mat_S = dl.as_backend_type(self.S).mat()
		self.S_csc = spsp.csc_matrix(mat_S.getValuesCSR()[::-1],shape=mat_S.size)
		mat_K = dl.as_backend_type(self.K).mat()
		self.K_csc = spsp.csc_matrix(mat_K.getValuesCSR()[::-1],shape=mat_K.size)
		
		mat_L = dl.as_backend_type(self.L).mat()
		L_csc = spsp.csc_matrix(mat_L.getValuesCSR()[::-1],shape=mat_L.size)
		self.L_chol = cholesky(L_csc)
		mat_M = dl.as_backend_type(self.M).mat()
		self.M_csc = spsp.csc_matrix(mat_M.getValuesCSR()[::-1],shape=mat_M.size)
		self.M_chol = cholesky(self.M_csc)
        
	def generate_vector(self, component = "ALL"):
		if component == "ALL":
			p = TimeDependentVector(self.simulation_times)
			p.initialize(self.M, 0)
			m = dl.Vector()
			self.prior.init_vector(m,0)  
			q = TimeDependentVector(self.simulation_times)
			q.initialize(self.M, 0)
			return [p, m, q]
		elif component == STATE:
			p = TimeDependentVector(self.simulation_times)
			p.initialize(self.M, 0)
			return p
		elif component == PARAMETER:
			m = dl.Vector()
			self.prior.init_vector(m,0)  
			return m
		elif component == ADJOINT:
			q = TimeDependentVector(self.simulation_times)
			q.initialize(self.M, 0)
			return q
		else:
			raise
			
	def cost(self,x):
		'''
		Evaluate cost at state x
		Input: x: [x[STATE], x[PARAMETER], x[ADJOINT]], x[STATE] TimeDependentVector, x[PARAMETER] log(mu_a) (Vector) 
		Output: [reg+misfit, reg, misfit], all components of cost 
		'''	
		reg = self.prior.cost(x[PARAMETER])
		misfit = self.misfit.cost(x)
		
		return [reg+misfit, reg, misfit]		
	
	def solveFwd(self,out,x):
		
		#Solve acoustic wave equation using Newmark scheme
		#Input: x: [~,x[PARAMETER],~]: x[PARAMETER]: initial pressure 
		#Output: pressure: TimeDependentVector 
		
		out.zero()

		p0_init = x[PARAMETER]
		
		# Initialize p,p',p'' for Newmark scheme
		pold = p0_init.vector().get_local()
		ptold = np.zeros(pold.shape)
		pttold = self.M_chol.solve_A(-1.0*(self.K_csc.dot(pold)))
		
		# Store initial condition in output
		pold_vec = dl.Vector()
		self.K.init_vector(pold_vec,0)
		pold_vec[:] = pold
		out.store(pold_vec,self.simulation_times[0])
		

		dtsqrd = (self.dt*self.dt)
		
		for t in self.simulation_times[1::]:
			Mrhs = (4/dtsqrd)*pold+(4/self.dt)*ptold+pttold
			Srhs = (2/self.dt)*pold+ptold
			pold_tmp = pold 
			
			# compute new p
			pold = self.L_chol.solve_A(((self.M_csc.dot(Mrhs))+(self.S_csc.dot(Srhs))))
			
			# compute new pt and new ptt
			pdiff = pold-pold_tmp
			ptold_tmp = ptold
			pttold_tmp = pttold 
			ptold = (2/self.dt)*pdiff-ptold_tmp
			pttold = (4/dtsqrd)*pdiff-(4/self.dt)*ptold_tmp-pttold_tmp
			pold_vec[:] = pold 
			
			out.store(pold_vec,t)
			
		
	def exportState(self, x, filename, varname):
		'''
		Export fwd solution, x[STATE] (TimeDependentVector) into paraview files
		'''
		out_file = dl.XDMFFile(self.Vh[STATE].mesh().mpi_comm(), filename)
		out_file.parameters["functions_share_mesh"] = True
		out_file.parameters["rewrite_function_mesh"] = False
		ufunc = dl.Function(self.Vh[STATE], name=varname)
		for t in self.simulation_times[0:]:
			x[STATE].retrieve(ufunc.vector(), t)
			out_file.write(ufunc, t)
			
	def exportData(self,x,filename): 
		'''
		Export data x (TimeDependentVector) into filename
		'''
		dVec = dl.Vector()
		self.misfit.B.init_vector(dVec,0)
		tmpData = np.zeros((self.misfit.ntargets*self.misfit.observation_times.shape[0],1))
		for i in range(0,self.misfit.observation_times.shape[0]):
			t = self.misfit.observation_times[i] 
			x.retrieve(dVec,t)
			tmpData[i*self.misfit.ntargets:(i+1)*self.misfit.ntargets,0] = dVec.get_local()
			
		np.savetxt(filename,tmpData)
		
	def importData(self,x,filename):
		'''
		Import data from filename into x (TimeDependentVector) 
		'''
		dArray = np.loadtxt(filename)
		x.zero()
		
		dVec = dl.Vector()
		self.misfit.B.init_vector(dVec,0)
		
		for i in range(0,self.misfit.observation_times.shape[0]):
			t = self.misfit.observation_times[i]
			dVec.set_local(dArray[i*self.misfit.ntargets:(i+1)*self.misfit.ntargets])
			x.store(dVec,t)

def convertTDV_to_Array(D_data_hat,nObs,observation_times):

	D_data_hat_array = np.zeros((nObs,len(observation_times)))
	t_indx = 0
	tmp_d = dl.Vector()
	misfit.B.init_vector(tmp_d,0)
	for times in observation_times:
		D_data_hat.retrieve(tmp_d,times)
		D_data_hat_array[:,t_indx] = tmp_d.get_local()
		t_indx = t_indx+1
	D_data_hat_array = D_data_hat_array.T
	D_data_hat_array = D_data_hat_array.ravel()
	
	return D_data_hat_array
	
def samplePrior(prior,Vh_param,pr_samps_file,nsamples):
	
	noise = dl.Vector()
	prior.init_vector(noise,"noise")
	m_samp = dl.Vector()
	prior.init_vector(m_samp, 0)
	
	m_samp_Fun = dl.Function(Vh_param,name="sample_prior")
	
	with dl.XDMFFile(mesh.mpi_comm(), pr_samps_file) as fid: 
		fid.parameters["functions_share_mesh"] = True
		fid.parameters["rewrite_function_mesh"] = False
		for i in range(0,nsamples):
			parRandom.normal(1.0,noise)
			prior.sample(noise,m_samp)
			m_samp_Fun.vector().set_local(np.exp(m_samp.get_local()))
			fid.write(m_samp_Fun,i)	
			
	return m_samp_Fun	
	
	
def samplePrior_noFile(prior,Vh_param): 
	noise = dl.Vector()
	prior.init_vector(noise,"noise")
	m_samp = dl.Vector()
	prior.init_vector(m_samp, 0)
	
	m_samp_Fun = dl.Function(Vh_param,name="sample_prior")
	
	parRandom.normal(1.0,noise)
	prior.sample(noise,m_samp)
	m_samp_Fun.vector().set_local((m_samp.get_local()))
	
	return m_samp_Fun

if __name__ == "__main__":
	'''
	Parse arguments 
	'''
	parser = argparse.ArgumentParser(description="PAI")
	parser.add_argument('--plots',default=1,type=int,help="Plot figures")
	parser.add_argument('--data_file',default='None',type=str,help="File containing synthetic data")
	parser.add_argument('--mesh_size',default=100,type=int,help="Mesh dimension is x by x, please input 130,150,170 or 200")
	args = parser.parse_args()

	plots_on = args.plots
	data_file = args.data_file
	mesh_dim = args.mesh_size
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
	Vh_param = Vh 
	ndofs_param = ndofs
	
	'''
	Set up time discretization
	'''
	if rank == 0: 
		print("Number of dof for state: {0}".format(ndofs))
		print("Number of dof for one parameter: {0}".format(ndofs_param))
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
	pr_mean = dl.interpolate(dl.Constant(-4.0),Vh_param)
	
	prior = BiLaplacianPrior(Vh_param, delta_prior, gamma_prior, mean=pr_mean.vector(),robin_bc=True)
	
	# Sample from the prior 
	sample_pr = True
	np.random.seed(42)
	if sample_pr:
		pr_samps_file = 'Plots/PAI/prior_samples.xdmf'
		nsamples = 15
		s_prior = samplePrior(prior,Vh_param,pr_samps_file,nsamples)
	
	m_hat = dl.Function(Vh_param)
	m_hat.vector().set_local((s_prior.vector().get_local()))
	c = dl.plot(m_hat)
	plt.colorbar(c)
	plt.show()
	
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
	#target_end_y = 2.9
	dxObs = 0.2
	targets_x = np.arange(target_start,target_end+0.5*dxObs,dxObs)
	#targets_y = np.arange(target_start,target_end_y+0.5*dxObs,dxObs)
	#targets = np.zeros((2*targets_x.shape[0]+2*targets_y.shape[0],2))
	targets = np.zeros((50,2))
	
	for i in range(0,targets_x.shape[0]): 
		targets[i,0] = targets_x[i]
		targets[i,1] = 3.0
		
	for i in range(0,targets_x.shape[0]):
		targets[i+targets_x.shape[0],0] = targets_x[i]
		targets[i+targets_x.shape[0],1] = 0.0
		
	#for i in range(0,targets_y.shape[0]):
	#	targets[i+2*targets_x.shape[0],0] = 0.0
	#	targets[i+2*targets_x.shape[0],1] = targets_y[i]
		
	#for i in range(0,targets_y.shape[0]):
	#	targets[i+2*targets_x.shape[0]+targets_y.shape[0],0] = 5.0
	#	targets[i+2*targets_x.shape[0]+targets_y.shape[0],1] = targets_y[i]
		
	nObs = targets.shape[0]
	
					
	misfit = SpaceTimePointwiseStateObservation(Vh,observation_times,targets)
	if rank == 0: 
		print('number of spatial observations: {0}'.format(targets.shape[0]))
		print('Total number of observations: {0}', targets.shape[0]*observation_times.shape[0])
		print('observation time step: {0}', dt_obs,sep)
		print('starting ob time: {0}', t1, sep)
		print('ending ob time: {0}', observation_times[-1],sep)
		print('number of observation times: {0}', observation_times.shape[0])
	#dsaklfjsal
		
	'''
	Setup model
	'''
	problem = AcousticWave(mesh,[Vh,Vh_param,Vh],prior,misfit,simulation_times)
	
	
	''' 
	Synthesize (or load in) data
	'''
	if rank == 0:
		print(sep,"Synthesizing data",sep) 
			
	noise = dl.Vector()
	prior.init_vector(noise,"noise")
	parRandom.normal(1., noise)
	
	p0_true = dl.Function(Vh_param)
	p0_true.vector().set_local(m_hat.vector().get_local())
	p_state = problem.generate_vector(STATE)
	q_state = problem.generate_vector(ADJOINT)
	x_state = [p_state,p0_true,q_state] 
		 
	start_time = time.time()
	problem.solveFwd(x_state[STATE],x_state)
	print("--- %s seconds ---" % (time.time() - start_time))

	problem.exportState(x_state, 'Plots/PAI/pressure.xdmf','pressure')
	misfit.observe(x_state,misfit.d) # apply observation operator 
	d_true = misfit.d
	max_d = misfit.d.norm("linf","linf")
	noise_std_dev = rel_noise*max_d # noise stdev
	misfit.noise_variance = noise_std_dev*noise_std_dev  
	

	print('observation times',observation_times*1e6)
	if plots_on == 1:
		# Visualize pressure data 
		tmp_d = dl.Vector()
		misfit.B.init_vector(tmp_d,0)
		d_2d = np.zeros((nObs,len(observation_times)))
		t_indx = 0
		for times in observation_times:
			misfit.d.retrieve(tmp_d,times)
			d_2d[:,t_indx] = tmp_d.get_local()
			t_indx = t_indx+1
		d_2d = d_2d.T
		print (sep, 'nobs is: ', nObs) 
		XX,YY = np.meshgrid(np.arange(1,nObs+1),(observation_times)*1e6)
		fig,ax = plt.subplots(1,1)
		color_map = plt.cm.get_cmap('YlGnBu_r')
		reserversed_color_map = color_map.reversed()
		cp = ax.contourf(XX,YY,d_2d,100,cmap=color_map)
		print('shape of xx is: ', d_2d.shape)
		fig.colorbar(cp)
		plt.title('Noisy pressure readings at target points',fontsize=30)
		plt.xlabel('Target number',fontsize=30)
		plt.ylabel('Time (in microseconds)',fontsize=30)
		plt.xticks(fontsize=30)
		plt.yticks(fontsize=30)
		plt.show()
		sys.exit()
		
	# Create fwd map matrix, used as observation operator in PAI.py
	
	samp_fwds = ndofs_param
	PTO_map = np.zeros((nObs*len(observation_times),ndofs_param))
	p0_i = dl.Function(Vh_param)
	i_vec = np.zeros(ndofs_param)
	for i in range(0,samp_fwds): 
		print("--- computing %d-th forward ---" % (i))
		i_vec[i] = 1
		p0_i.vector().set_local(i_vec)
		
		p_state = problem.generate_vector(STATE)
		q_state = problem.generate_vector(ADJOINT)
		x_state = [p_state,p0_i,q_state] 
		 
		start_time = time.time()
		problem.solveFwd(x_state[STATE],x_state)

		misfit.observe(x_state,misfit.d) # apply observation operator   
		
		data_i = convertTDV_to_Array(misfit.d,nObs,observation_times)
		
		PTO_map[:,i] = data_i
		i_vec[i] = 0
		
	np.save("ObsOperator/PTO_Acoustic_original.npy", PTO_map) 
	
	

	
	
	
	
			
		

    

   
   
   
   
      
   

