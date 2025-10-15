# TumorTwin HGG Example

This code uses a slightly altered verion of TumorTwin sent to me by Anirban.

## Some notes:
 - This code uses data from the first patient in the data in the TumorTwin repository.
 - `self.voxel_coords` is going to be a useful attribute.
 - Initially, should probably just treat the proliferation rate as uncertain (and keep the other parameters at their calibrated values). Use a truncated KL expansion for the prior, and probably centre it quite close to the calibrated (scalar) value of the coefficient. Use ~10 modes.
 - Could start by considering the scalar-valued parametrisation and making sure it actually runs.

## TODO:
 - Figure out how to plot the (spatially varying) proliferation rate.
 - Figure out a way to handle model failures. This could involve using an adaptive solver and determining when it has taken a very large number of steps, or using a fixed-step solver and checking the outputs for physical realism.
 - Figure out how to parametrise the prior for the spatially varying proliferation rate. Could use the SPDE representation of the Matern field (although solving the linear system might take a while, and this would need to be combined with some of the dimension reduction ideas). The eigendecomposition of the covariance matrix is too expensive to compute. 