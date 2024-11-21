from matplotlib import pyplot as plt
import numpy as np
from scipy import stats


class Banana():
    """Form of distribution from TransportMaps package.
    """

    def __init__(self, a, b, mu, cov):

        self.base_dist = stats.multivariate_normal(mu, cov)
        self.B_inv = lambda x: np.array([x[0] / a, 
                                         a * (x[1] + b*(x[0]**2 + a**2))])

    def pdf(self, x):
        return self.base_dist.pdf(self.B_inv(x))


if __name__ == "__main__":
    
    a = 1.0
    b = 1.0
    mu = np.zeros(2)
    cov = np.array([[1.0, 0.9],
                    [0.9, 1.0]])

    banana = Banana(a, b, mu, cov)

    nx = 100

    x0s = np.linspace(-4, 4, nx)
    x1s = np.linspace(-9, 3, nx)

    coords = [[x0, x1] for x1 in x1s for x0 in x0s]
    density = [banana.pdf(c) for c in coords]

    density = np.reshape(density, (nx, nx))

    plt.contour(x0s, x1s, density)
    plt.show()