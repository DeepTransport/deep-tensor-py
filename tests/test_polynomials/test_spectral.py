import unittest

import torch

import deep_tensor as dt


torch.manual_seed(0)


class TestSpectralPolynomials(unittest.TestCase):

    def test_orthogonality(self):
        """Confirms that the product of the basis2node and node2basis 
        matrices is the identity matrix (see Cui and Dolgov 2022, 
        Appendix A).
        """

        polynomials: list[dt.Spectral] = [
            dt.Chebyshev1st(order=20),
            dt.Chebyshev2ndUnweighted(order=20),
            dt.Fourier(order=20),
            dt.Legendre(order=20)
        ]

        for poly in polynomials:
            with self.subTest(poly=poly):

                Id = poly.basis2node @ poly.node2basis 
                max_difference = (torch.eye(poly.cardinality) - Id).abs().max()

                self.assertTrue(max_difference < 1e-8)

        return