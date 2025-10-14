# TumorTwin HGG Example

This code uses a slightly altered verion of TumorTwin sent to me by Anirban.

Some notes:
 - This code uses data from the first patient in the data in the TumorTwin repository.
 - `self.voxel_coords` is going to be a useful attribute.
 - Initially, should probably just treat the proliferation rate as uncertain (and keep the other parameters at their calibrated values). Use a truncated KL expansion for the prior, and probably centre it quite close to the calibrated (scalar) value of the coefficient. Use ~10 modes.