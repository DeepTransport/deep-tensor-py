import dolfin as dl
import hippylib as hl
import ufl

mesh = dl.UnitSquareMesh(64, 64)

Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

alpha = 3
beta = 1.2 
u0 = dl.Expression(
    "1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t", 
    alpha=alpha, 
    beta=beta, 
    t=0,
    element=Vh.ufl_element()
)

u_1 = dl.interpolate(u0, Vh)

dt = 0.3  # Timestep

u = dl.TrialFunction(Vh)
v = dl.TestFunction(Vh)
f = dl.Constant(beta - 2.0 - 2.0 * alpha)

a = u * v * ufl.dx + dt * ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v)) * ufl.dx 
L = (u_1 + dt * f) * v * ufl.dx

A = dl.assemble(a)

u = dl.Function(Vh)
T = 10.0
t = dt 

while t <= T:

    b = dl.assemble(L)  # assemble L with updated u_1
    u0.t = t  # what is this?

    dl.solve(A, u.vector(), b)
    t += dt 
    
    u_1.assign(u)

    u_array = u.vector()[:]
    print(u_array.min())
    print(u_array.max())

    dl.plot(u)
    from matplotlib import pyplot as plt
    plt.show()