import numpy as np

from tt_cross import Tensor, tt_cross
from old.banana import Banana


a = 1.0
b = 1.0
mu = np.zeros(2)
cov = np.array([[1.0, 0.9],
                [0.9, 1.0]])

banana = Banana(a, b, mu, cov)

nx = 20

grids = [
    np.linspace(-4, 4, nx),
    np.linspace(-9, 3, nx)
]

t = Tensor(func=banana.pdf, ndim=2, grids=grids)

col_index_sets = [
    np.arange(5, dtype=int),
    np.arange(5, dtype=int)
]

tt_cross(t, col_index_sets)