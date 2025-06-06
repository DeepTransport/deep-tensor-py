import abc
from typing import Tuple

from torch import Tensor


class MarginalEstimator(abc.ABC):
    r"""A sample-based estimator of a marginal density."""

    @abc.abstractmethod
    def eval_pdf(self, xs: Tensor) -> Tensor:
        return
    
    @abc.abstractmethod
    def eval_potential(self, xs: Tensor) -> Tensor:
        return
    
    @abc.abstractmethod
    def eval_cdf(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        return
    
    @abc.abstractmethod
    def invert_cdf(self, zs: Tensor) -> Tensor:
        return