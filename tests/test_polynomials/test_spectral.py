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
            dt.Chebyshev2nd(order=20),
            dt.Fourier(order=20),
            dt.Legendre(order=20)
        ]

        for poly in polynomials:
            with self.subTest(poly=poly):
                Id = poly.node2basis @ poly.basis2node
                Id_true = torch.eye(poly.cardinality)
                self.assertTrue((Id_true - Id).abs().max() < 1e-4)

        return


if __name__ == "__main__":
    unittest.main()