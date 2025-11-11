from typing import Callable
import math

import torch 
from torch import Tensor


PISTON_LBS = torch.tensor([30, 0.005, 0.002, 1000,  90_000, 290, 340])
PISTON_UBS = torch.tensor([60, 0.020, 0.010, 5000, 110_000, 296, 360])

BOREHOLE_LBS = torch.tensor([0.05,    100,  63_070,  990,  63.1, 700, 1120,   9855])
BOREHOLE_UBS = torch.tensor([0.15, 50_000, 115_600, 1110, 116.0, 820, 1680, 12_045])

OTL_CIRCUIT_LBS = torch.tensor([ 50, 25, 0.5, 1.2, 0.25,  50])
OTL_CIRCUIT_UBS = torch.tensor([150, 70, 3.0, 2.5, 1.20, 300])

WING_WEIGHT_LBS = torch.tensor([150, 220,  6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025])
WING_WEIGHT_UBS = torch.tensor([200, 300, 10,  10, 45, 1.0, 0.18, 6.0, 2500, 0.080])


def rescale(ls: Tensor, lbs: float | Tensor, ubs: float | Tensor) -> Tensor:
    ls = 0.5 * (ls + 1.0)
    ls = lbs + ls * (ubs-lbs)
    return ls


def ackley(ls: Tensor) -> Tensor:
    ls = rescale(ls, -32.768, 32.768)
    fxs = (
        - 20.0 * torch.exp(-0.2 * ls.square().mean(dim=1).sqrt())
        - torch.exp(torch.cos(2*torch.pi*ls).mean(dim=1)) + 20 + math.exp(1.0)
    )
    return fxs


def alpine(ls: Tensor) -> Tensor:
    # Mistake in Strossner et al.: should have ls[:, :1].
    ls = rescale(ls, -10.0, 10.0)
    fxs = torch.sum(torch.abs(ls * torch.sin(ls) + 0.1 * ls), dim=1)
    return fxs


def dixon(ls: Tensor) -> Tensor:
    ls = rescale(ls, -10.0, 10.0)
    fls = (ls[:, 0] - 1.0) ** 2 
    for i in range(1, 7):
        fls += (i + 1) * (2.0 * ls[:, i] ** 2 - ls[:, i-1]) ** 2
    return fls


def exponential(ls: Tensor) -> Tensor:
    fls = -torch.exp(-0.5 * torch.sum(ls**2, dim=1))
    return fls


def griewank(ls: Tensor) -> Tensor:
    # Mistakes in Strosser et al.: missing sqrt(), and division by i 
    # moved outside cosinse function
    ls = rescale(ls, -600.0, 600.0)
    is_ = torch.arange(1, 8)
    fls = (
        torch.sum((ls ** 2), dim=1) / 4000.0
        - torch.prod(torch.cos(ls) / is_, dim=1)
        + 1
    )
    return fls


def michaelwicz(ls: Tensor) -> Tensor:
    ls = rescale(ls, 0.0, torch.pi)
    is_ = torch.arange(1, 8)
    fls = -torch.sum(
        torch.sin(ls) * torch.sin(is_ * ls**2 / torch.pi) ** 20, 
        dim=1
    )
    return fls


def piston(ls: Tensor) -> Tensor:
    # Mistake in Strossner et al.: A not inside sqrt() when computing V.
    ls = rescale(ls, PISTON_LBS, PISTON_UBS)
    M, S, V0, k, P0, Ta, T0 = ls.T 
    A = P0*S + 19.62*M - (k*V0)/S
    V = (S/(2*k)) * (torch.sqrt(A**2 + 4*k*(P0*V0/T0)*Ta) - A)
    fls = 2 * torch.pi * torch.sqrt(M / (k + S**2 * (P0*V0/T0) * (Ta / V**2)))
    return fls


def qing(ls: Tensor) -> Tensor:
    ls = rescale(ls, 0.0, 500.0)
    is_ = torch.arange(1, 8)
    fls = torch.sum((ls**2 - is_)**2, dim=1)
    return fls


def rastrigin(ls: Tensor) -> Tensor:
    ls = rescale(ls, -5.12, 5.12)
    fls = 70 + torch.sum(ls**2 - 10.0 * torch.cos(2.0*torch.pi*ls), dim=1)
    return fls


def rosenbrock(ls: Tensor) -> Tensor:
    ls = rescale(ls, -2.048, 2.048)
    fls = torch.zeros((ls.shape[0],))
    for i in range(6):
        fls += (
            100.0 * torch.square(ls[:, i+1] - ls[:, i]**2)
            + torch.square(1.0 - ls[:, i])
        )
    return fls


def schaffer(ls: Tensor) -> Tensor:
    ls = rescale(ls, -100.0, 100.0)
    fls = torch.zeros((ls.shape[0],))
    for i in range(6):
        ls_sq = ls[:, i]**2 + ls[:, i+1]**2
        fls += (
            0.5 
            + (torch.sin(torch.sqrt(ls_sq)).square() - 0.5) 
            / (1.0 + 0.001 * ls_sq).square()
        )
    return fls


def schwefel(ls: Tensor) -> Tensor:
    ls = rescale(ls, -500.0, 500.0)
    return 2932.8803 - torch.sum(ls*torch.sin(torch.abs(ls)**0.5), dim=1)


def borehole(ls: Tensor) -> Tensor:
    ls = rescale(ls, BOREHOLE_LBS, BOREHOLE_UBS)
    rw, r, Tu, Hu, Tl, Hl, L, Kw = ls.T
    frac1 = 2 * torch.pi * Tu * (Hu-Hl)
    frac2a = 2*L*Tu / (torch.log(r/rw)*rw**2*Kw)
    frac2b = Tu / Tl
    frac2 = torch.log(r/rw) * (1 + frac2a + frac2b)
    fls = frac1 / frac2
    return fls


def otl_circuit(ls: Tensor) -> Tensor:
    ls = rescale(ls, OTL_CIRCUIT_LBS, OTL_CIRCUIT_UBS)
    b1, b2, f, c1, c2, beta = ls.T
    x = beta*(c2+9.0)
    t1 = (12*b2/(b1+b2) + 0.74) * x / (x+f)
    t2 = (11.35*f) / (x+f)
    t3 = (0.74*f*x) / ((x+f)*c1)
    Vm = t1 + t2 + t3
    return Vm


def robot_arm(ls: Tensor) -> Tensor:
    # Mistake in Strosser et al.: Indexing issues in sum.
    ts = rescale(ls[:, :4], 0.0, 2.0*torch.pi)
    Ls = rescale(ls[:, 4:], 0.0, 1.0)
    u = torch.zeros((ls.shape[0],))
    v = torch.zeros((ls.shape[0],))
    for i in range(4):
        theta_i = torch.sum(ts[:, :i+1], dim=1)
        u += Ls[:, i] * torch.cos(theta_i)
        v += Ls[:, i] * torch.sin(theta_i)
    return torch.sqrt(u**2 + v**2)


def wing_weight(ls: Tensor) -> Tensor:
    ls = rescale(ls, WING_WEIGHT_LBS, WING_WEIGHT_UBS)
    Sw, Wf, A, delta, q, lamb, tc, Nz, Wd, Wp = ls.T
    delta *= (torch.pi / 180) # degrees to radians
    fls = (
        0.036
        * Sw**0.758
        * Wf**0.0035
        * (A / torch.cos(delta)**2)**0.6
        * q**0.006 
        * lamb**0.04
        * (100*tc/torch.cos(delta))**-0.3
        * (Nz*Wd)**0.49
        + Sw*Wp
    )
    return fls


def friedman(ls: Tensor) -> Tensor:
    ls = rescale(ls, 0.0, 1.0)
    fls = (
        10.0 * torch.sin(torch.pi*ls[:, 0]*ls[:, 1])
        + 20.0 * (ls[:, 2]-0.5)**2
        + 10.0 * ls[:, 3]
        + 5.0 * ls[:, 4]
    )
    return fls


def gnl(ls: Tensor) -> Tensor:
    ls = rescale(ls, 0.0, 1.0)
    fls = (
        torch.exp(torch.sin((0.9*(ls[:, 0]+0.48))**10.0)) 
        + ls[:, 1]*ls[:, 2] 
        + ls[:, 3]
    )
    return fls


def dnp_8d(ls: Tensor) -> Tensor:
    ls = rescale(ls, 0.0, 1.0)
    t1 = 4.0 * (ls[:, 0] - 2.0 + 8.0*ls[:, 1] - 8.0*ls[:, 1]**2)**2
    t2 = (3.0 - 4.0*ls[:, 1])**2
    t3 = 16.0*torch.sqrt(ls[:, 2]+1.0) * (2.0*ls[:, 2] - 1.0)**2
    t4 = torch.zeros((ls.shape[0],))
    for i in range(3, 8):
        t4 += (i+1) * torch.log(1.0+torch.sum(ls[:, 2:i+1], dim=1))
    return t1 + t2 + t3 + t4


def dnp_exp(ls: Tensor) -> Tensor:
    ls = rescale(ls, 0.0, 1.0)
    fls = 100 * (
        torch.exp(-2.0/(ls[:, 0]**1.75)) 
        + torch.exp(-2.0/(ls[:, 1]**1.5))
        + torch.exp(-2.0/(ls[:, 2]**1.25))
    )
    return fls


def normalise_cs(cs: Tensor, dim: int, b: float, h: float) -> Tensor:
    c_norm = (dim**h / b) * cs.abs().sum()
    return cs / c_norm


def build_genz_1(dim: int) -> Callable:
    # Oscillatory Genz

    w = torch.rand(1)
    cs = torch.rand(dim)
    cs = normalise_cs(cs, dim, b=284.6, h=1.5)
    
    def genz_1(ls: Tensor) -> Tensor:
        fls = torch.cos(2*torch.pi*w + torch.sum(cs*0.5*(ls+1), dim=1))
        return fls
    
    return genz_1


def build_genz_2(dim: int) -> Callable:
    # Corner peak Genz

    cs = torch.rand(dim)
    cs = normalise_cs(cs, dim, b=185.0, h=2.0)
    
    def genz_2(ls: Tensor) -> Tensor:
        fls = (1.0 + torch.sum(cs*0.5*(ls+1), dim=1)) ** -(dim-1)
        return fls 
    
    return genz_2

def build_genz_3(dim: int) -> Callable:
    # Continuous Genz

    ws = torch.rand(dim)
    cs = torch.rand(dim)
    cs = normalise_cs(cs, dim, b=2040.0, h=2.0)

    def genz_3(ls: Tensor) -> Tensor:
        fls = torch.exp(-torch.sum(cs**2 * torch.abs(0.5*(ls+1) - ws), dim=1))
        return fls

    return genz_3


FUNCTIONS = {
    r"Ackley": (ackley, 7),
    r"Alpine": (alpine, 7),
    r"Dixon": (dixon, 7),
    r"Exponential": (exponential, 7),
    r"Griewank": (griewank, 7),
    r"Michaelwicz": (michaelwicz, 7),
    r"Piston": (piston, 7),
    r"Qing": (qing, 7),
    r"Rastrigin": (rastrigin, 7),
    r"Rosenbrock": (rosenbrock, 7),
    r"Schaffer": (schaffer, 7),
    r"Schwefel": (schwefel, 7),
    r"Borehole": (borehole, 8),
    r"OTL Circuit": (otl_circuit, 6),
    r"Robot Arm": (robot_arm, 8),
    r"Wing Weight": (wing_weight, 10),
    r"Friedman": (friedman, 5),
    r"G\&L": (gnl, 6),
    r"D\&P 8D": (dnp_8d, 8),
    r"D\&P Exp": (dnp_exp, 3)
}