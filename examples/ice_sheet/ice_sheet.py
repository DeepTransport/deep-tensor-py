import copy

from dolfin import *
import torch
import numpy as np
from ufl import nabla_div
import matplotlib.pyplot as plt

# Take logs of Robin coefficient when looking at the ISMIP stuff

# Define domain parameters
length = 10000.0
height = 1000.0                         # Height of domain
angle = 0.1 * torch.pi / 180.0          # Slope (radians)

grav = 9.81                             # Gravity
rho = 910.0                             # Ice density
sConst = Constant(1.e-8)
scale_const = Constant(1.0e-10)
N = 3.0                                 # Rheology parameter?
Aconst = Constant(2.140373 * 1.0e7)     # Ice flow parameter


class PeriodicBoundary(SubDomain):
    
    def inside(self, x, on_boundary):
        return bool(near(x[0], 0) and on_boundary)
        # return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
    
    def map(self, x, y):
        y[0] = x[0] - length
        y[1] = x[1]
        return

class Top(SubDomain):
   
   def inside(self, x, on_boundary):
       return near(x[1], height)

class Bottom(SubDomain):

    def inside(self, x, on_boundary):
        return near(x[1], 0)

top = Top()
sides = PeriodicBoundary()
bottom = Bottom()

# 1. Build domain and mark subregions
mesh = RectangleMesh(Point(0, 0), Point(length, height), 40, 4)  # AdB: can vary discretisation here
boundary_mesh = BoundaryMesh(mesh, "exterior", True)

# 2. Define function spaces
P2 = VectorElement(family="Lagrange", cell=mesh.ufl_cell(), degree=2)  # x and y, 2=order (need higher order for velocity)
P1 = FiniteElement(family="Lagrange", cell=mesh.ufl_cell(), degree=1)
TH = P2 * P1

VQ = FunctionSpace(mesh, TH, constrained_domain=sides)   # periodic product space for velocity and pressure
VP = FunctionSpace(mesh, 'Lagrange', 1)
Vh = [VQ, VP, VQ]

# 3 Boundary Conditions and Forcing Term
# AdB: driven by gravity. ice modelled on flat surface and give it a 
# nonzero foring term (sin of angle). Equivalent to solving on tilted domain.

# Define true basal sliding coefficient
beta_true = Expression(
    cpp_code="std::log(1001.0 + 1000.0 * sin(x[0] * 2.0 * pi / length))", 
    element=VP.ufl_element(), 
    length=length
)

f = Expression( 
    cpp_code=("rho * grav * sin(angle)", "-rho * grav * cos(angle)"),
    element=VQ.sub(0).ufl_element(), 
    angle=angle, 
    grav=grav, 
    rho=rho
)

beta_interp = interpolate(beta_true, VP)

File("btrue.pvd") << beta_interp

bc = DirichletBC(VQ.sub(0).sub(1), Constant(0.0), bottom)

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

Gamma_B = Bottom()
Gamma_B.mark(boundary_markers,1)
ds = Measure(
    integral_type="ds", 
    domain=mesh, 
    subdomain_data=boundary_markers
)

# Normal vector and tangential operators
normal = FacetNormal(mesh)
def Tang(u, normal): 
    return (u - outer(normal, normal) * u)

# Define strains
def epsilon(u):
    return Constant(0.5) * (nabla_grad(u) + nabla_grad(u).T)

# 4 Linear Problem for initial guess
(uh, ph) = TrialFunctions(VQ)
(vh, qh) = TestFunctions(VQ)

# Linear version of Stokes
a_linear = (Aconst * inner(epsilon(uh), epsilon(vh)) * dx 
            - div(vh) * ph * dx 
            - qh * div(uh) * dx 
            + inner(exp(beta_true) * Tang(uh, normal), Tang(vh, normal)) * ds(1))
L_linear = inner(f, vh) * dx


vq = Function(VQ)
solve(a_linear == L_linear, vq, bc) # initial guess for nonlinear solve... 
# if verifyFwd:
#     u_dir = Function(VQ)
#     solve(a_linear_dir == L_linear, u_dir, bc)

(uh_lin, ph_lin) = vq.split(deepcopy=True)

uh = uh_lin

ufile_pvd = File("linear_velocity.pvd")
ufile_pvd << uh_lin

# 5 Define exponential parameter field and Energy functional, and 1st and 2nd variations

n = interpolate(Constant(float(N)), VP) 

if N==1:
    Aconst_NL = 1./Aconst  # pre-flow prefactor
else:
    Aconst_NL = 10.0 ** -16.0

(u, p) = vq.split()
(v, q) = TestFunctions(VQ)
(w, r) = TrialFunctions(VQ)


# AdB: should be in line with Georg's stuff
def Energy(u):
    normEu12 = 0.5*inner(epsilon(u),epsilon(u)) + sConst
    return scale_const*(Aconst_NL**(-1./n)*((2.*n)/(1.+n))*(normEu12**((1. + n)/(2.*n)))*dx - \
           inner(f,u)*dx + Constant(.5)*inner(exp(beta_true)*Tang(u,normal), \
           Tang(u,normal)) * ds(1))

def Gradient(u,v,p,q):  # u, test func for u, pressure, test func for pressure
    normEu12 = 0.5*inner(epsilon(u),epsilon(u)) + sConst
    return scale_const*(Aconst_NL**(-1./n)*((normEu12**((1.- n)/(2.*n)))*inner(epsilon(u),epsilon(v)))*dx - \
           inner(f,v)*dx + inner(exp(beta_true)*Tang(u,normal), Tang(v,normal)) * ds(1) - \
           p*nabla_div(v)*dx -q*nabla_div(u)*dx)


# Next variations
def Hessian(u,v,w,p,q,r): # p=pressure, q=test func for pressure, r=direction you are applying it in.
    normEu12 = 0.5*inner(epsilon(u),epsilon(u)) + sConst
    return scale_const * (Aconst_NL ** (-1.0 / n)
                          * (((1.0-n)/(2.0*n))*(normEu12**((1.-3.*n)/(2.*n))* \
           inner(epsilon(u),epsilon(w))*inner(epsilon(v),epsilon(u))) + \
           ((normEu12**((1.-n)/(2.0*n)))*inner(epsilon(w),epsilon(v))))*dx + \
           inner(exp(beta_true)*Tang(w,normal), Tang(v,normal)) * ds(1) - \
           r*nabla_div(v)*dx - q*nabla_div(w)*dx)

LS = True # Use Line Search
rtol = 1e-8
atol = 1e-8
max_iter = 100
maxbackit = 50
c_armijo = 1e-4

dvq = Function(VQ)
dvq.assign(vq)

J = Energy(u)
G = Gradient(u,v,p,q)
H = Hessian(u,v,w,p,q,r)

Ju0 = assemble(J)
Jn = copy.deepcopy(Ju0) # For line search
g0 = assemble(G) # initial gradient
g0_norm = g0.norm("l2")

for i in range(1,max_iter+1):
    [Hn, gn] = assemble_system(H, G, bc)
    Hn.init_vector(dvq.vector(),1)

    solve(Hn,dvq.vector(),gn)
    gn_norm = gn.norm("l2")
    dvq_gn = -dvq.vector().inner(gn)

    alpha = 1
    print(i)
    if LS:
        vq_back = vq.copy(deepcopy=True)
        bk_converged = False
        
        for j in range(maxbackit):
            vq.vector().axpy(-alpha,dvq.vector())
            (u,p) = vq.split()
            
            J = Energy(u)
            Jnext = assemble(J)
            
            if Jnext < Jn + abs(alpha*c_armijo*dvq_gn):
                Jn = Jnext
                bk_converged = True
                break
            
            alpha = alpha/2.
            vq.assign(vq_back)
        
        if not bk_converged:
            vq.assign(vq_back)
            print('max backtracking')
            break
    else:
        vq.vector().axpy(-alpha,dvq.vector())
        (u,p) = vq.split()
        J = Energy(u)
        Jn = assemble(J)
        
    print(gn_norm)
    G = Gradient(u,v,p,q)
    H = Hessian(u,v,w,p,q,r)

# (uh, ph) = u_init.split(deepcopy=True)

(u, p) = vq.split(deepcopy=True)
ufile_pvd = File("nonlinear_velocity.pvd")
ufile_pvd << u

nx = 32
ny = 32
mesh = UnitSquareMesh(nx,ny)
Vh = FunctionSpace(mesh, "CG", 1)

uh = Function(Vh)
u_hat = TestFunction(Vh)
u_tilde = TrialFunction(Vh)

# nb.plot(mesh)
# print "dim(Vh) = ", Vh.dim()




f = Constant(1.)
k1 = Constant(0.05)
k2 = Constant(1.)

Pi = Constant(.5)*(k1 + k2*uh*uh)*inner(nabla_grad(uh), nabla_grad(uh))*dx - f*uh*dx

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

u_0 = Expression("0", degree=2) 
bc = DirichletBC(Vh,u_0, Boundary() )


grad = (k2*uh*u_hat)*inner(nabla_grad(uh), nabla_grad(uh))*dx + \
       (k1 + k2*uh*uh)*inner(nabla_grad(uh), nabla_grad(u_hat))*dx - f*u_hat*dx

u0 = interpolate(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)",degree=2), Vh)

n_eps = 32
eps = 1e-2 * 2.0 ** -torch.arange(n_eps)
err_grad = torch.zeros(n_eps)

uh.assign(u0)
pi0 = assemble(Pi)
grad0 = assemble(grad)

dir = Function(Vh)
dir.vector().set_local(np.random.randn(Vh.dim()))
bc.apply(dir.vector())
dir_grad0 = grad0.inner(dir.vector())

for i in range(n_eps):
    uh.assign(u0)
    uh.vector().axpy(eps[i], dir.vector()) #uh = uh + eps[i]*dir
    piplus = assemble(Pi)
    err_grad[i] = abs( (piplus - pi0)/eps[i] - dir_grad0 )

plt.figure()    
plt.loglog(eps, err_grad, "-ob")
plt.loglog(eps, (.5*err_grad[0]/eps[0])*eps, "-.k")
plt.title("Finite difference check of the first variation (gradient)")
plt.xlabel("eps")
plt.ylabel("Error grad")
# plt.legend(["Error Grad", "First Order"], "upper left")

# plt.show()

H = k2*u_tilde*u_hat*inner(nabla_grad(uh), nabla_grad(uh))*dx + \
     Constant(2.)*(k2*uh*u_hat)*inner(nabla_grad(u_tilde), nabla_grad(uh))*dx + \
     Constant(2.)*k2*u_tilde*uh*inner(nabla_grad(uh), nabla_grad(u_hat))*dx + \
     (k1 + k2*uh*uh)*inner(nabla_grad(u_tilde), nabla_grad(u_hat))*dx

uh.assign(u0)
H_0 = assemble(H)
err_H = torch.zeros(n_eps)
for i in range(n_eps):
    uh.assign(u0)
    uh.vector().axpy(eps[i], dir.vector())
    grad_plus = assemble(grad)
    diff_grad = (grad_plus - grad0)
    diff_grad *= 1/eps[i]
    H_0dir = H_0 * dir.vector()
    err_H[i] = (diff_grad - H_0dir).norm("l2")
    
plt.figure()    
plt.loglog(eps, err_H, "-ob")
plt.loglog(eps, (.5*err_H[0]/eps[0])*eps, "-.k")
plt.title("Finite difference check of the second variation (Hessian)")
plt.xlabel("eps")
plt.ylabel("Error Hessian")
# plt.legend(["Error Hessian", "First Order"], "upper left")

plt.show()