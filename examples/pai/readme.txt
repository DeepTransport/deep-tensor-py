Contains code to evaluate the PTO map: p(t_i,x_j) = B G(m)
as well as its Jacobian
  
# Optic fwd problem is in G(m): 
G(m) = mu_a(m)*phi(m), phi(m) obtained via solving optic PDE:
mu_a(m) = exp(m), kappa(mu_a) = 1/(2*(mu_a+0.2*mu_s)), mu_s fixed constant 
mu_a phi - \div(kappa(mu_a) \grad(phi)) = 0 in Omega
(1/pi)*phi - 0.5*kappa(mu_a)*\grad(phi)*n = s on dOmega
p0 = mu_a*phi

# Acoustic problem is in the operator B:
  B: passed in matrix, takes p_0, outputs p(t_i,x_j) at 5 observation times, 50 x_js via solution of acoustic wave equation: 
  p_tt - c^2 \Delta p = 0 in Omega x [0,T]
  p(t = 0) = p0 in Omega
  p_t(t = 0) = 0 in Omega
  Boundary conditions are chosen to approximate an absorbing boundary
  
applying the PTO map, as well as computing the Jacobian + some sanity checks are in PAI_run.py

ObsOperator contains a B that gets passed in

Note: If you want to change the mesh or observation points/times, you will need to adjust it in Acoustic_original.py and re-run the code to generate a new B


Code works with dolfin version 2019.2.0.13.dev0
