import abc
import warnings

from torch import Tensor 


class Basis1D(abc.ABC, object):
    """The parent class for all one-dimensional bases.
    """

    _nodes: Tensor
    _mass_R: Tensor

    @property
    @abc.abstractmethod
    def _domain(self) -> Tensor:
        """The (local) domain of the basis."""
        pass

    @property
    @abc.abstractmethod
    def _constant_weight(self) -> bool:
        """Returns whether the weighting function of the basis is 
        constant for all values of l.
        """
        pass

    @property
    def _cardinality(self) -> int:
        """The number of basis functions associated with the basis."""
        return self._nodes.numel()
    
    @abc.abstractmethod
    def _eval_basis(self, ls: Tensor) -> Tensor:
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
        pass
    
    @abc.abstractmethod
    def _eval_basis_deriv(self, ls: Tensor) -> Tensor:
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
        pass 
    
    @abc.abstractmethod
    def _eval_measure(self, ls: Tensor) -> Tensor:
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
        pass
    
    @abc.abstractmethod
    def _eval_measure_deriv(self, ls: Tensor) -> Tensor:
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
        pass
    
    @abc.abstractmethod 
    def _eval_log_measure(self, ls: Tensor) -> Tensor:
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
        pass 

    @abc.abstractmethod
    def _eval_log_measure_deriv(self, ls: Tensor) -> Tensor:
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
        pass

    @abc.abstractmethod
    def _sample_measure(self, n: int) -> Tensor:
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
        pass
    
    def _out_domain(self, ls: Tensor) -> Tensor:
        """Returns a boolean mask that indicates whether each of a set
        of points is outside the local domain of the basis.        
        """
        return (ls < self._domain[0]) | (ls > self._domain[1])
    
    def _check_in_domain(self, ls: Tensor) -> None:
        """Checks whether a set of points are inside the domain, and 
        issues a warning if not.
        """
        if (outside := self._out_domain(ls)).any():
            msg = f"{outside.sum()} points are outside the domain."
            # warnings.warn(msg) # TODO: add logger here instead.
        return
    
    def _check_eval_dims(self, coeffs: Tensor, ls: Tensor) -> None:
        """Checks that the arguments of 'eval_radon', 'eval', 
        'eval_radon_deriv' and 'eval_deriv' have the correct 
        dimensions.
        """
        if ls.ndim != 1:
            msg = "'ls' must be a vector."
            raise Exception(msg)
        if coeffs.shape[0] != self._cardinality:
            msg = ("Coefficient vector must have the same cardinality "
                   + "as the basis.")
            raise Exception(msg)
        return
    
    def _eval_radon(self, coeffs: Tensor, ls: Tensor) -> Tensor:
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
        self._check_eval_dims(coeffs, ls)
        basis_vals = self._eval_basis(ls)
        fls = basis_vals @ coeffs
        return fls
        
    def _eval_radon_deriv(self, coeffs: Tensor, ls: Tensor) -> Tensor:
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
        self._check_eval_dims(coeffs, ls)
        deriv_vals = self._eval_basis_deriv(ls)
        gradfls = deriv_vals @ coeffs
        return gradfls

    def _eval(self, coeffs: Tensor, ls: Tensor) -> Tensor:
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
        self._check_eval_dims(coeffs, ls)
        fls = self._eval_radon(coeffs, ls)
        wls = self._eval_measure(ls)
        return fls * wls[:, None]
        
    def _eval_deriv(self, coeffs: Tensor, ls: Tensor) -> Tensor:
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
        self._check_eval_dims(coeffs, ls)
        # Compute first term of product rule
        dpdls = self._eval_basis_deriv(ls)
        wls = self._eval_measure(ls)[:, None]
        gradfwls = dpdls @ coeffs * wls
        # Compute second term of product rule
        if not self._constant_weight:
            basis_vals = self._eval_basis(ls)
            gradwls = self._eval_measure_deriv(ls)[:, None]
            gradfwls = gradfwls + (basis_vals @ coeffs) * gradwls
        return gradfwls