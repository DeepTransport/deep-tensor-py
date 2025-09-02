import unittest

import deep_tensor as dt


class TestOptions(unittest.TestCase):

    def test_tt_options(self):

        tt_options = dt.TTOptions(tt_method="FIXED_RANK")
        self.assertTrue(tt_options.tt_method == "fixed_rank")

        # Check that the TT method is adjusted to "fixed_rank" (i.e., 
        # no enrichment) if the kick rank is equal to 0
        tt_options = dt.TTOptions(kick_rank=0)
        self.assertTrue(tt_options.tt_method == "fixed_rank")

        # Check that unknown methods throw an error
        self.assertRaises(ValueError, dt.TTOptions, tt_method="unknown")
        self.assertRaises(ValueError, dt.TTOptions, int_method="unknown")
        return