import abc

import torch


class Basis1D(abc.ABC, object):
    """The parent class for all one-dimensional bases.
    """

    @property
    @abc.abstractmethod
    def domain(self) -> torch.Tensor:
        """The domain of the basis."""
        pass

    @property
    @abc.abstractmethod
    def nodes(self) -> torch.Tensor:
        """The nodes associated with the basis. For spectral 
        polynomials, these are the collocation points.
        """
        pass

    @property
    @abc.abstractmethod
    def constant_weight(self) -> bool:
        """Returns whether the weighting function of the basis is 
        constant for all x.
        """
        pass

    @property 
    @abc.abstractmethod
    def mass_R(self) -> torch.Tensor:
        """Cholesky factor of the matrix containing the (weighted) 
        inner products of each pair of basis functions over the 
        reference domain.
        """
        return 
    
    @property 
    @abc.abstractmethod
    def int_W(self) -> torch.Tensor: 
        """TODO: write."""
        return

    @abc.abstractmethod
    def eval_basis(self, ls: torch.Tensor) -> torch.Tensor:
        """Evaluates the (normalised) one-dimensional basis at a given 
        set of points.
        
        Parameters 
        ----------
        ls:
            An n-dimensional vector of points (on the interval [-1, 1])
            at which to evaluate the basis functions.
        
        Returns
        -------
        basis_vals:
            An n * d matrix containing the values of each basis 
            function evaluated at each point. Element (i, j) contains 
            the value of the jth basis function evaluated at the ith 
            element of ls.
        
        """
        return
    
    @abc.abstractmethod
    def eval_basis_deriv(self, ls: torch.Tensor) -> torch.Tensor:
        """Evaluates the derivative of each (normalised) basis function
        at a given set of points.
        
        Parameters
        ----------
        ls: 
            An n-dimensional vector of points (on the interval [-1, 1])
            at which to evaluate the derivative of each basis function.
        
        Returns
        -------
        deriv_vals:
            An n * d matrix containing the derivative of each basis 
            function evaluated at each point. Element (i, j) contains 
            the derivative of the jth basis function evaluated at the
            ith element of ls.

        """
        return 
    
    @abc.abstractmethod
    def eval_measure(self, ls: torch.Tensor) -> torch.Tensor:
        """Evaluates the (normalised) weighting function at a given set
        of points.
        
        Parameters
        ----------
        ls: 
            An n-dimensional vector of points at which to evaluate the 
            weighting function.

        Returns
        -------
        :
            The value of the weighting function evaluated at each 
            location.
                
        """
        return
    
    @abc.abstractmethod
    def eval_measure_deriv(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: write docstring."""
        return
    
    @abc.abstractmethod 
    def eval_log_measure(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the logarithm of the (normalised) weighting 
        function at a given set of points.
        
        Parameters
        ----------
        x: 
            The locations at which to evaluate the logarithm of the 
            weighting function.

        Returns
        -------
        :
            The logarithm of the weighting function evaluated at each 
            location.
        
        """
        return 

    @abc.abstractmethod
    def eval_log_measure_deriv(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the gradient of the logarithm of the (normalised)
        weighting function at a given set of points.
        
        Parameters
        ----------
        x:
            The locations at which to evaluate the gradient of the 
            logarithm of the weighting function.

        Returns
        -------
        :
            The logarithm of the gradient of the weighting function 
            evaluated at each location.
        
        """
        return

    @abc.abstractmethod
    def sample_measure(self, n: int) -> torch.Tensor:
        """Generates a set of samples from the weighting measure
        corresponding to the one-dimensional basis.
        
        Parameters
        ----------
        n: 
            Number of samples to generate.

        Returns
        -------
        :
            The generated samples.
        
        """
        return

    @property
    def has_bounded_domain(self) -> bool:
        """Whether the domain of the basis is bounded."""
        return not torch.any(torch.isinf(self.domain))

    @property
    def cardinality(self) -> int:
        """The number of nodes associated with the basis."""
        return self.nodes.numel()

    def in_domain(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a boolean mask that indicates whether each of a set
        of points is contained within the domain of the basis.
        
        Parameters
        ----------
        x: 
            A set of points.

        Returns
        -------
        :
            A boolean mask that indicates whether each x is contained 
            within the domain.
        
        """
        return (x >= self.domain[0]) & (x <= self.domain[1])
    
    def out_domain(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a boolean mask that indicates whether each of a set
        of points is outside the domain of the basis.
        
        Parameters
        ----------
        x: 
            A set of points.

        Returns
        -------
        :
            A boolean mask that indicates whether each x is outside the 
            domain.
        
        """
        return (x < self.domain[0]) | (x > self.domain[1])
    
    def eval(
        self, 
        coeffs: torch.Tensor, 
        xs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the product of approximated function and the 
        weighting function at a given vector of points.
        
        Parameters 
        ----------
        coeffs:
            The coefficients associated with each basis function.
        xs:
            Points at which to evaluate the approximated function.

        Returns
        -------
        :
            The values of the product of the approximated function 
            and the weighting function at each point.
        
        """
        
        func_vals = self.eval_radon(coeffs, xs)
        weights = self.eval_measure(xs)
        return func_vals * weights
        
    def eval_deriv(
        self, 
        coeffs: torch.Tensor, 
        xs: torch.Tensor
    ) -> torch.Tensor:
        """Not sure what this is for yet."""
        
        deriv_vals = torch.zeros((xs.shape, coeffs.shape[1]))
        points_in_domain = self.in_domain(xs)

        if not torch.any(points_in_domain):
            return deriv_vals
        
        basis_vals = self.eval_basis_deriv(xs[points_in_domain])
        deriv_vals[points_in_domain] = basis_vals @ coeffs * self.eval_measure(xs[points_in_domain])

        if not self.constant_weight:
            basis_vals = self.eval_basis(xs[points_in_domain])
            deriv_vals[points_in_domain] = deriv_vals[points_in_domain] + basis_vals @ coeffs * self.eval_measure_deriv(xs[points_in_domain])

        return deriv_vals
 
    def eval_radon(
        self, 
        coeffs: torch.Tensor, 
        xs: torch.Tensor
    ) -> torch.Tensor:
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
        xs:
            A vector of points at which to evaluate the approximated 
            function.

        Returns
        -------
        func_vals:
            A matrix containing the values of each basis function 
            evaluated at each point. Element (i, j) contains the value 
            of the jth basis function evaluated at the ith value of x.
        
        """

        func_vals = torch.zeros((xs.numel(), coeffs.shape[1]))

        points_in_domain = self.in_domain(xs)
        if not torch.any(points_in_domain):
            return func_vals
        
        basis_vals = self.eval_basis(xs[points_in_domain])
        func_vals[points_in_domain] = basis_vals @ coeffs
        return func_vals
        
    def eval_radon_deriv(
        self, 
        coeffs: torch.Tensor, 
        xs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the derivative of the approximated function at
        a set of points.

        Parameters 
        ----------
        coeffs:
            The coefficients associated with each basis function.
        xs:
            Points at which to evaluate the approximated function.

        Returns
        -------
        deriv:
            The values of the derivative of the function at each point.
        
        """

        deriv_vals = torch.zeros((xs.numel(), coeffs.shape[1]))
        points_in_domain = self.in_domain(xs)

        if not torch.any(points_in_domain):
            return deriv_vals 
        
        basis_vals = self.eval_basis_deriv(xs[points_in_domain])
        deriv_vals[points_in_domain] = basis_vals @ coeffs
        return deriv_vals
    
    def mass_r(self, interp_w: torch.Tensor) -> torch.Tensor:
        """Evaluates the product of the upper Cholesky factor of the 
        mass matrix and a set of nodal values or coefficients.
        """
        return self.mass_R @ interp_w
    
    def evaluate_integral(self, interp_w: torch.Tensor) -> torch.Tensor:
        """Evaluates the one-dimensional integral."""
        return self.int_W @ interp_w