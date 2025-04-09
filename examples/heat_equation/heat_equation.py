from dolfin import *
import matplotlib.pyplot as plt

"""From https://fenics-handson.readthedocs.io/en/latest/heat/doc.html
"""


def create_timestep_solver(get_data, dsN, theta, u_old, u_new):
    """Prepare timestep solver by theta-scheme for given
    function get_data(t) returning data (f(t), g(t)), given
    solution u_old at time t and unknown u_new at time t + dt.
    Return a solve function taking (t, dt).
    """

    # Initialize coefficients
    f_n, g_n = get_data(0)
    f_np1, g_np1 = get_data(0)
    idt = Constant(0)

    # Extract function space
    V = u_new.function_space()

    # Prepare weak formulation
    u, v = TrialFunction(V), TestFunction(V)
    theta = Constant(theta)
    F = ( idt*(u - u_old)*v*dx
        + inner(grad(theta*u + (1-theta)*u_old), grad(v))*dx
        - (theta*f_np1 + (1-theta)*f_n)*v*dx
        - (theta*g_np1 + (1-theta)*g_n)*v*dsN
    )
    a, L = lhs(F), rhs(F)

    def solve_(t, dt):
        """Update problem data to interval (t, t+dt) and
        run the solver"""

        # Update coefficients to current t, dt
        get_data(t, (f_n, g_n))
        get_data(t+dt, (f_np1, g_np1))
        idt.assign(1/dt)

        # Push log level
        old_level = get_log_level()
        warning = LogLevel.WARNING # if cpp.__version__ > '2017.2.0' else WARNING
        set_log_level(warning)

        # Run the solver
        solve(a == L, u_new)

        # Pop log level
        set_log_level(old_level)

    return solve_


def timestepping(V, dsN, theta, T, dt, u_0, get_data):
    """Perform timestepping using theta-scheme with
    final time T, timestep dt, initial datum u_0 and
    function get_data(t) returning (f(t), g(t))"""

    # Initialize solution function
    u = Function(V)

    # Prepare solver for computing time step
    solver = create_timestep_solver(get_data, dsN, theta, u, u)

    # Set initial condition
    u.interpolate(u_0)

    # Open plot window
    fig = init_plot()

    # Print table header
    print("{:10s} | {:10s} | {:10s}".format("t", "dt", "energy"))

    # Perform timestepping
    t = 0
    while t < T:

        # Report some numbers
        energy = assemble(u*dx)
        print("{:10.4f} | {:10.4f} | {:#10.4g}".format(t, dt, energy))

        # Perform time step
        solver(t, dt)
        t += dt

        # Update plot
        update_plot(fig, u)


def timestepping_adaptive(V, dsN, theta, T, tol, u_0, get_data):
    """Perform adaptive timestepping using theta-scheme with
    final time T, tolerance tol, initial datum u_0 and
    function get_data(t) returning (f(t), g(t))"""

    # Initialize needed functions
    u_n = Function(V)
    u_np1_low = Function(V)
    u_np1_high = Function(V)

    # Prepare solvers for computing tentative time steps
    solver_low = create_timestep_solver(get_data, dsN, theta, u_n, u_np1_low)
    solver_high_1 = create_timestep_solver(get_data, dsN, theta, u_n, u_np1_high)
    solver_high_2 = create_timestep_solver(get_data, dsN, theta, u_np1_high, u_np1_high)

    # Initial time step; the value does not really matter
    dt = T/2

    # Set initial conditions
    u_n.interpolate(u_0)

    # Open plot window
    fig = init_plot()

    # Print table header
    print("{:10s} | {:10s} | {:10s}".format("t", "dt", "energy"))

    # Perform timestepping
    t = 0
    while t < T:

        # Report some numbers
        energy = assemble(u_n*dx)
        print("{:10.4f} | {:10.4f} | {:#10.4g}".format(t, dt, energy))

        # Compute tentative time steps
        solver_low(t, dt)
        solver_high_1(t, dt/2)
        solver_high_2(t+dt, dt/2)

        # Compute error estimate and new timestep
        est = compute_est(theta, u_np1_low, u_np1_high)
        dt_new = compute_new_dt(theta, est, tol, dt)

        if est > tol:
            # Tolerance not met; repeat the step with new timestep
            dt = dt_new
            continue

        # Move to next time step
        u_n.vector()[:] = u_np1_high.vector()
        t += dt
        dt = dt_new

        # Update plot
        update_plot(fig, u_n)


def compute_est(theta, u_L, u_H):
    """Return error estimate by Richardson extrapolation"""
    p = 2 if theta == 0.5 else 1
    est = sqrt(assemble((u_L - u_H)**2*dx)) / (2**p - 1)
    return est


def compute_new_dt(theta, est, tol, dt):
    """Return new time step"""
    p = 2 if theta == 0.5 else 1
    rho = 0.9
    dt_new = dt * ( rho * tol / est )**(1/p)
    return dt_new


def init_plot():
    """Open plot window and return its figure object"""
    fig = plt.figure()
    fig.show()
    return fig


def update_plot(fig, u, zlims=(0, 2)):
    """Plot u in 3D warp mode with colorbar into figure fig;
    use zlims as limits on z-axis"""
    fig.clear()
    p = plot(u, mode="warp")
    if p is None:
        return
    fig.colorbar(p)
    fig.gca().set_zlim(zlims)
    fig.canvas.draw()


def create_function_space():
    """Return (arbitrary) H^1 conforming function space on
    unit square domain"""
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "P", 1)
    return V


def create_surface_measure_left(mesh):
    """Return surface measure on the left boundary of unit
    square"""
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0)
    facets = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    Left().mark(facets, 1)
    ds_left = Measure("ds", mesh, subdomain_data=facets, subdomain_id=1)
    return ds_left


def get_data_2(t, result=None):
    """Create or update data for Task 2"""
    f, g = result or (Constant(0), Constant(0))
    f.assign(0)
    g.assign(0)
    return f, g


def get_data_3(t, result=None):
    """Create or update data for Task 3"""
    f, g = result or (Constant(0), Constant(0))
    f.assign(1)
    g.assign(0)
    return f, g


def get_data_4(t, result=None):
    """Create or update data for Task 4"""
    f, g = result or (Constant(0), Constant(0))
    f.assign(2-t)
    g.assign(t)
    return f, g


def get_data_5(t, result=None):
    """Create or update data for Task 5 and Task 6"""
    f, g = result or (Constant(0), Constant(0))
    f.assign(0)
    g.assign(max(0, 1-t)/2)
    return f, g


if __name__ == '__main__':
    # Common data
    V = create_function_space()
    ds_left = create_surface_measure_left(V.mesh())
    T = 2
    u_0 = Expression("x[0]", degree=1)

    # Run all problems
    theta = 1
    print("Task 2, theta =", theta)
    dt = 0.1
    timestepping(V, ds_left, theta, T, dt, u_0, get_data_2)

    theta = 1/2
    print("Task 2, theta =", theta)
    dt = 0.1
    timestepping(V, ds_left, theta, T, dt, u_0, get_data_2)

    theta = 0
    print("Task 2, theta =", theta)
    dt = 0.1
    timestepping(V, ds_left, theta, T, dt, u_0, get_data_2)

    theta = 1
    print("Task 3, theta =", theta)
    dt = 0.1
    timestepping(V, ds_left, theta, T, dt, u_0, get_data_3)

    theta = 1
    print("Task 4, theta =", theta)
    dt = 0.1
    timestepping(V, ds_left, theta, T, dt, u_0, get_data_4)

    theta = 1
    print("Task 5, theta =", theta)
    dt = 0.1
    timestepping(V, ds_left, theta, T, dt, u_0, get_data_5)

    theta = 1
    print("Task 6, theta =", theta)
    tol = 1e-3
    timestepping_adaptive(V, ds_left, theta, T, tol, u_0, get_data_5)

    theta = 1/2
    print("Task 6, theta =", theta)
    tol = 1e-3
    timestepping_adaptive(V, ds_left, theta, T, tol, u_0, get_data_5)

    # Hold plots before quitting
    plt.show()