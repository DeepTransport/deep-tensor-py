## TODO
 - Eventually it might be good to have a *core* object that holds information about the interpolation basis, the coefficient tensor, etc.
 - Could use a generator for sample generation in future?
 - Remove calls to `torch.linalg.inv` (solve linear systems instead).