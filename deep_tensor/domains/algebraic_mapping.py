from typing import Tuple

from torch import Tensor

from .mapped_domain import MappedDomain
from ..constants import EPS
from ..tools import check_finite


class AlgebraicMapping(MappedDomain):
    r"""Mapping from an unbounded domain to $(-1, 1)$.
    
    This class provides a mapping from an unbounded domain, 
    $(-\infty, \infty)$, to a bounded domain, $(-1, 1)$. This mapping 
    is of the form 
    $$x \mapsto \frac{x/s}{\sqrt{1 + (x/s)^{2}}},$$
    where $s$ is a scale parameter.

    Parameters
    ----------
    scale:
        The scale parameter, $s$.
    
    """

    def approx2local(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        xs = xs / self.scale
        ts = 1.0 + xs.square()
        ls = xs * ts.pow(-0.5)
        dldxs = ts.pow(-1.5) / self.scale
        check_finite(ls)
        check_finite(dldxs)
        return ls, dldxs
    
    def approx2local_log_density(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        xs = xs / self.scale 
        ts = 1.0 + xs.square()
        logdldxs = -1.5 * ts.log() - self.scale.log()
        logd2ldx2s = (-3.0 / self.scale) * (xs / ts)
        check_finite(logdldxs)
        check_finite(logd2ldx2s)
        return logdldxs, logd2ldx2s
    
    def local2approx(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        ls = ls.clamp(-1.0+EPS, 1.0-EPS)
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS 
        xs = self.scale * ls * ts.pow(-0.5)
        dxdls = self.scale * ts.pow(-1.5)
        check_finite(xs)
        check_finite(dxdls)
        return xs, dxdls 
    
    def local2approx_log_density(self, ls: Tensor) -> Tuple[Tensor, Tensor]:
        ts = 1.0 - ls.square()
        ts[ts < EPS] = EPS 
        logdxdls = -1.5 * ts.log() + self.scale.log()
        logd2xdl2 = 3.0 * (ls / ts)
        check_finite(logdxdls)
        check_finite(logd2xdl2)
        return logdxdls, logd2xdl2