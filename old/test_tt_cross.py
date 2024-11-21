import numpy as np

from tt_cross import Grid, tt_cross

np.random.seed(0)


n = 3
A = np.random.rand(n, n, n)

def A_func(inds):
    inds = [int(i) for i in inds]
    return A[*inds]

grids = [np.arange(n, dtype=int)] * 3

grid = Grid(grids)

# These implicitly define the ranks of the cores as well
col_index_sets = [
    # [(0, 0, 0), (1, 1, 1), (2, 2, 2)], 
    [(0, 0), (1, 1), (2, 2)], 
    [(0, ), (1, ), (2, )],
    [[]]
]

tt = tt_cross(A_func, grid, col_index_sets)

print(tt[1, 1, 1])
print(A[1, 1, 1])

print(tt[0, -1, -1])
print(A[0, -1, -1])