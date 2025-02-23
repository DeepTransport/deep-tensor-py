import abc

import torch
from torch import Tensor 


class Basis1D(abc.ABC, object):
    """The parent class for all one-dimensional bases.
    """

    @property
    @abc.abstractmethod
    def domain(self) -> Tensor:
        """The (local) domain of the basis."""
        pass

    @property
    @abc.abstractmethod
    def nodes(self) -> Tensor:
        """The nodes associated with the basis. For spectral 
        polynomials, these are the collocation points.
        """
        pass

    @property
    @abc.abstractmethod
    def constant_weight(self) -> bool:
        """Returns whether the weighting function of the basis is 
        constant for all values of l.
        """
        pass

    @property 
    @abc.abstractmethod
    def mass_R(self) -> Tensor:
        """Cholesky factor of the matrix containing the (weighted) 
        inner products of each pair of basis functions over the 
        local domain.
        """
        return 
    
    @property 
    @abc.abstractmethod
    def int_W(self) -> Tensor: 
        """Given a set of polynomial coefficients, this operator 
        returns the values of the integrated function at each 
        collocation point.
        """
        return

    @abc.abstractmethod
    def eval_basis(self, ls: Tensor) -> Tensor:
        """Evaluates the (normalised) one-dimensional basis at a given 
        set of points.
        
        Parameters 
        ----------
        ls:
            An n-dimensional vector of points in the local domain at 
            which to evaluate the basis functions.
        
        Returns
        -------
        ps:
            An n * d matrix containing the values of each basis 
            function evaluated at each point. Element (i, j) contains 
            the value of the jth basis function evaluated at the ith 
            element of ls.
        
        """
        return
    
    @abc.abstractmethod
    def eval_basis_deriv(self, ls: Tensor) -> Tensor:
        """Evaluates the derivative of each (normalised) basis function
        at a given set of points.
        
        Parameters
        ----------
        ls: 
            An n-dimensional vector of points in the local domain at 
            which to evaluate the derivative of each basis function.
        
        Returns
        -------
        dpdls:
            An n * d matrix containing the derivative of each basis 
            function evaluated at each point. Element (i, j) contains 
            the derivative of the jth basis function evaluated at the
            ith element of ls.

        """
        return 
    
    @abc.abstractmethod
    def eval_measure(self, ls: Tensor) -> Tensor:
        """Evaluates the (normalised) weighting function at a given set
        of points.
        
        Parameters
        ----------
        ls: 
            An n-dimensional vector of points in the local domain at 
            which to evaluate the weighting function.

        Returns
        -------
        wls:
            An n-dimensional vector containing the value of the 
            weighting function evaluated at each point in ls.
                
        """
        return
    
    @abc.abstractmethod
    def eval_measure_deriv(self, ls: Tensor) -> Tensor:
        """Evaluates the gradient of the (normalised) weighting 
        function at a given set of points.
        
        Parameters
        ----------
        ls:
            An n-dimensional vector of points in the local domain at 
            which to evaluate the gradient of the weighting function.
        
        Returns
        -------
        gradwls:
            An n-dimensional vector containing the value of the 
            gradient of the weighting function evaluated at each point
            in ls.

        """
        return
    
    @abc.abstractmethod 
    def eval_log_measure(self, ls: Tensor) -> Tensor:
        """Evaluates the logarithm of the (normalised) weighting 
        function at a given set of points.
        
        Parameters
        ----------
        ls: 
            An n-dimensional vector of points in the local domain at
            which to evaluate the logarithm of the weighting function.

        Returns
        -------
        logwls:
            An n-dimensional vector containing the value of the 
            logarithm of the weighting function evaluated at each point
            in ls.
        
        """
        return 

    @abc.abstractmethod
    def eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
        """Evaluates the gradient of the logarithm of the (normalised) 
        weighting function at a given set of points.
        
        Parameters
        ----------
        ls:
            An n-dimensional vector of points in the local domain at
            which to evaluate the logarithm of the gradient of the 
            weighting function.

        Returns
        -------
        loggradwls:
            An n-dimensional vector containing the gradient of the 
            logarithm of the weighting function evaluated at each point
            in ls.
        
        """
        return

    @abc.abstractmethod
    def sample_measure(self, n: int) -> Tensor:
        """Generates a set of samples from the weighting measure
        corresponding to the one-dimensional basis.
        
        Parameters
        ----------
        n: 
            The number of samples to generate.

        Returns
        -------
        ls:
            An n-dimensional vector containing the generated samples.
        
        """
        return

    @property
    def has_bounded_domain(self) -> bool:
        """Whether the domain of the basis is bounded."""
        return self.domain.isinf().any()

    @property
    def cardinality(self) -> int:
        """The number of nodes associated with the basis."""
        return self.nodes.numel()

    def in_domain(self, ls: Tensor) -> Tensor:
        """Returns a boolean mask that indicates whether each of a set
        of points is contained within the local domain of the basis.
        
        Parameters
        ----------
        ls: 
            An n-dimensional vector.

        Returns
        -------
        inside:
            An n-dimensional boolean mask that indicates whether each 
            element of ls is contained within the local domain.
        
        """
        return (ls >= self.domain[0]) & (ls <= self.domain[1])
    
    def out_domain(self, ls: Tensor) -> Tensor:
        """Returns a boolean mask that indicates whether each of a set
        of points is outside the local domain of the basis.
        
        Parameters
        ----------
        ls: 
            A set of points.

        Returns
        -------
        outside:
            A boolean mask that indicates whether each element of ls 
            is outside the local domain.
        
        """
        return (ls < self.domain[0]) | (ls > self.domain[1])
    
    def eval_radon(self, coeffs: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the approximated function at a given vector of 
        points.
        
        Parameters 
        ----------
        coeffs:
            A matrix containing the nodal values (piecewise 
            polynomials) or coefficients (spectral polynomials) 
            associated with each basis function. Element (i, j) 
            contains the jth coefficient associated with the ith basis 
            function.
        ls:
            A vector of points at which to evaluate the approximated 
            function.

        Returns
        -------
        fls:
            A matrix containing the values of the approximated function 
            evaluated at each point in ls. Element (i, j) contains the 
            value of the function evaluated at the ith element of ls 
            using the jth set of coefficients (i.e., the jth column of 
            coeffs).
        
        """

        fls = torch.zeros((ls.numel(), coeffs.shape[1]))
        
        if not torch.any(inside := self.in_domain(ls)):
            return fls
        
        basis_vals = self.eval_basis(ls[inside])
        fls[inside] = basis_vals @ coeffs
        return fls
        
    def eval_radon_deriv(self, coeffs: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the derivative of the approximated function at
        a set of points.

        Parameters 
        ----------
        coeffs:
            A matrix containing the nodal values (piecewise 
            polynomials) or coefficients (spectral polynomials) 
            associated with each basis function. Element (i, j) 
            contains the jth coefficient associated with the ith basis 
            function.
        ls:
            A vector of points at which to evaluate the derivative of 
            the approximated function.

        Returns
        -------
        gradfls:
            The values of the derivative of the function at each point.
        
        """

        gradfls = torch.zeros((ls.numel(), coeffs.shape[1]))

        if not torch.any(inside := self.in_domain(ls)):
            return gradfls 
        
        deriv_vals = self.eval_basis_deriv(ls[inside])
        gradfls[inside] = deriv_vals @ coeffs
        return gradfls

    def eval(self, coeffs: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the product of approximated function and the 
        weighting function at a given vector of points.
        
        Parameters 
        ----------
        coeffs:
            A matrix containing the nodal values (piecewise 
            polynomials) or coefficients (spectral polynomials) 
            associated with each basis function. Element (i, j) 
            contains the jth coefficient associated with the ith basis 
            function.
        ls:
            An n-dimensional vector containing points (within the local
            domain) at which to evaluate the approximated function.

        Returns
        -------
        fwls:
            A matrix containing the values of the product of the 
            approximated function and the weighting function evaluated 
            at each point in ls. Element (i, j) contains the product of 
            the weighting function and the approximated function 
            (using the jth set of coefficients) evaluated at the ith 
            element of ls.
        
        """
        fls = self.eval_radon(coeffs, ls)
        wls = self.eval_measure(ls).reshape(-1, 1)
        return fls * wls
        
    def eval_deriv(self, coeffs: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the gradient of the product of the approximated 
        function and the weighting function at a given vector of 
        points.

        Parameters
        ----------
        coeffs: 
            A matrix containing the nodal values (piecewise 
            polynomials) or coefficients (spectral polynomials) 
            associated with each basis function. Element (i, j) 
            contains the jth coefficient associated with the ith basis 
            function.
        ls:
            An n-dimensional vector containing points (within the local
            domain) at which to evaluate the approximated function.

        Returns
        -------
        gradfwls:
            A matrix containing the values of the gradient of the 
            product of the approximated function and the weighting 
            function evaluated at each point in ls. Element (i, j) 
            contains the product of the weighting function and the 
            approximated function (using the jth set of coefficients) 
            evaluated at the ith element of ls.
        
        """
        
        gradfwls = torch.zeros((ls.numel(), coeffs.shape[1]))

        if not torch.any(inside := self.in_domain(ls)):
            return gradfwls
        
        # Compute first term of product rule
        dpdls = self.eval_basis_deriv(ls[inside])
        wls = self.eval_measure(ls[inside])
        gradfwls[inside] = dpdls @ coeffs * wls[:, None]

        # Compute second term of product rule
        if not self.constant_weight:
            basis_vals = self.eval_basis(ls[inside])
            gradwls = self.eval_measure_deriv(ls[inside])
            gradfwls[inside] += (basis_vals @ coeffs) * gradwls[:, None]

        return gradfwls
 
    def mass_r(self, interp_w: Tensor) -> Tensor:
        """Evaluates the product of the upper Cholesky factor of the 
        mass matrix and a set of nodal values or coefficients.
        """
        return self.mass_R @ interp_w
    
    def evaluate_integral(self, interp_w: Tensor) -> Tensor:
        """Evaluates the one-dimensional integral."""
        return self.int_W @ interp_w