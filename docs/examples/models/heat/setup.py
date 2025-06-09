import dolfin as dl
import hippylib as hl
import torch 
from torch import Tensor

from .solver import HeatSolver, HeatSolverROM
from .reduced_order_modelling import compute_pod_basis, compute_rom_matrices
from ..priors import ProcessConvolutionPrior


# Not working...
def _build_obs_operator(mesh: dl.Mesh, targets: Tensor) -> Tensor:

    tree = dl.BoundingBoxTree()
    tree.build(mesh)
    cell_to_vertices = mesh.topology()(mesh.topology().dim(), 0)

    B = torch.zeros((targets.shape[0], mesh.num_vertices()))

    for i, t in enumerate(targets):
        
        point = dl.Point(*t)
        cell = tree.compute_first_entity_collision(point)
        vertices = cell_to_vertices(cell).tolist()
        cs = mesh.coordinates()[vertices]

        denom = ((cs[1][1] - cs[2][1]) * (cs[0][0] - cs[2][0]) + (cs[2][0] - cs[1][0]) * (cs[0][1] - cs[2][1]))
        w0 = ((cs[1][1] - cs[2][1]) * (t[0] - cs[2][0]) + (cs[2][0] - cs[1][0]) * (t[1] - cs[2][1])) / denom
        w1 = ((cs[2][1] - cs[0][1]) * (t[0] - cs[2][0]) + (cs[0][0] - cs[2][0]) * (t[1] - cs[2][1])) / denom
        w2 = 1.0 - w0 - w1
        
        B[i][vertices] = torch.tensor([w0, w1, w2])
    
    return B


def build_obs_operator(Vh: dl.FunctionSpace, xs: Tensor) -> Tensor:
    B = hl.assemblePointwiseObservation(Vh, xs)
    B = torch.from_numpy(B.array())
    return B


def setup_heat_problem():

    mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(3.0, 1.0), 96, 32)
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

    # Define prior
    xs = torch.tensor(Vh.tabulate_dof_coordinates())

    # Define locations of Gaussian white noise
    s0s = torch.linspace(3/18, 51/18, 9)
    s1s = torch.linspace(1/6, 5/6, 3)
    ss = torch.tensor([[s0, s1] for s0 in s0s for s1 in s1s])

    prior = ProcessConvolutionPrior(xs, ss)

    dt_reg = 0.25
    t_init = dt_reg
    t_final = 10.0
    obs_ind_0 = 3
    obs_dx = 4
        
    ts = torch.arange(t_init, t_final+0.5*dt_reg, dt_reg)

    xs_obs = torch.tensor([
        [0.4, 0.4], [0.4, 0.6], 
        [0.6, 0.4], [0.6, 0.6], 
        [2.4, 0.4], [2.4, 0.5], [2.4, 0.6], 
        [2.5, 0.4], [2.5, 0.5], [2.5, 0.6], 
        [2.6, 0.4], [2.6, 0.5], [2.6, 0.6]
    ])

    B = build_obs_operator(Vh, xs_obs)

    model = HeatSolver(
        mesh=mesh, 
        V=Vh, 
        prior=prior, 
        ts=ts, 
        xs_obs=xs_obs,
        obs_ind_0=obs_ind_0, 
        obs_dx=obs_dx,
        B=B
    )

    n_snapshots = 1000
    xi_samples = torch.randn((n_snapshots, model.prior.dim))
    snapshots = torch.hstack([model.solve(prior.transform(x)) for x in xi_samples])

    print("Computing POD basis...")
    m, V = compute_pod_basis(snapshots)
    K_rs = compute_rom_matrices(model.mesh, V, model.V)

    rom = HeatSolverROM(model, m, V, K_rs)

    return prior, model, rom