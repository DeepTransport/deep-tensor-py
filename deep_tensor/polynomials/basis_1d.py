import abc
import warnings

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
    def nodes(self) -> Tensor:
        return self._nodes 

    @nodes.setter 
    def nodes(self, value: Tensor) -> None:
        self._nodes = value 
        return

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
        pass 
    
    @property 
    @abc.abstractmethod
    def int_W(self) -> Tensor: 
        """Given a set of polynomial coefficients, this operator 
        returns the values of the integrated function at each 
        collocation point.
        """
        pass

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
        pass
    
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
        pass 
    
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
        pass
    
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
        pass
    
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
        pass 

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
        pass

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
        pass

    @property
    def has_bounded_domain(self) -> bool:
        """Whether the domain of the basis is bounded."""
        return bool(self.domain.isinf().any())

    @property
    def n_nodes(self) -> int:
        """The number of nodes associated with the basis."""
        return self.nodes.numel()

    @property
    def cardinality(self) -> int:
        """The number of basis functions associated with the basis."""
        return self.nodes.numel()
    
    def _out_domain(self, ls: Tensor) -> Tensor:
        """Returns a boolean mask that indicates whether each of a set
        of points is outside the local domain of the basis.        
        """
        return (ls < self.domain[0]) | (ls > self.domain[1])
    
    def _check_in_domain(self, ls: Tensor) -> None:
        """Checks whether a set of points are inside the domain, and 
        issues a warning if not.
        """
        if (outside := self._out_domain(ls)).any():
            msg = f"{outside.sum()} points are outside the domain."
            warnings.warn(msg)
        return
    
    def _check_eval_dims(self, coeffs: Tensor, ls: Tensor) -> None:
        """Checks that the arguments of 'eval_radon', 'eval', 
        'eval_radon_deriv' and 'eval_deriv' have the correct 
        dimensions.
        """
        if ls.ndim != 1:
            msg = "'ls' must be a vector."
            raise Exception(msg)
        if coeffs.shape[0] != self.cardinality:
            msg = ("Coefficient vector must have the same cardinality "
                   + "as the basis.")
            raise Exception(msg)
        return
    
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
        self._check_eval_dims(coeffs, ls)
        basis_vals = self.eval_basis(ls)
        fls = basis_vals @ coeffs
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
        self._check_eval_dims(coeffs, ls)
        deriv_vals = self.eval_basis_deriv(ls)
        gradfls = deriv_vals @ coeffs
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
        self._check_eval_dims(coeffs, ls)
        fls = self.eval_radon(coeffs, ls)
        wls = self.eval_measure(ls)
        return fls * wls[:, None]
        
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
        self._check_eval_dims(coeffs, ls)
        # Compute first term of product rule
        dpdls = self.eval_basis_deriv(ls)
        wls = self.eval_measure(ls)[:, None]
        gradfwls = dpdls @ coeffs * wls

        # Compute second term of product rule
        if not self.constant_weight:
            basis_vals = self.eval_basis(ls)
            gradwls = self.eval_measure_deriv(ls)[:, None]
            gradfwls += (basis_vals @ coeffs) * gradwls

        return gradfwls
 
    def mass_r(self, interp_w: Tensor) -> Tensor:
        """Evaluates the product of the upper Cholesky factor of the 
        mass matrix and a set of nodal values or coefficients.
        """
        return self.mass_R @ interp_w
    
    def evaluate_integral(self, interp_w: Tensor) -> Tensor:
        """Evaluates the one-dimensional integral."""
        return self.int_W @ interp_w