## TODO
 - Eventually it might be good to have a *core* object that holds information about the interpolation basis, the coefficient tensor, etc.
 - Could use a generator for sample generation in future?
 - Remove calls to `torch.linalg.inv` (solve linear systems instead).

## Some notation
 - `xs` refers to samples in the approximation domain. `zs` refers to samples in [0, 1] (generally evaluations of the CDF). `us` refers to samples from [-1, 1].