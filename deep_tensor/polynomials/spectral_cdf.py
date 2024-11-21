import abc
from typing import Tuple
import warnings

import torch

from .oned_cdf import OnedCDF
from ..constants import EPS


class SpectralCDF(OnedCDF, abc.ABC):
    """CDF class for spectral polynomials.

    Properties
    ----------
    sampling_nodes:
        A set of locations at which the basis function are evaluated.
    cdf_basis2node:
        A matrix containing the indefinite integrals of the product of 
        each basis function and the weighting function, evaluated at 
        each sampling node.

    """

    def __init__(self, **kwargs):
        
        OnedCDF.__init__(self, **kwargs)

        num_sampling_nodes = max(2*self.cardinality, 200)

        self.sampling_nodes = self.grid_measure(num_sampling_nodes)
        self.cdf_basis2node = self.eval_int_basis(self.sampling_nodes)
        return

    @property
    @abc.abstractmethod  # TODO: figure out whether this should go in OnedCDF or SpectralCDF.
    def cardinality(self) -> torch.Tensor:
        return
    
    @property 
    @abc.abstractmethod 
    def node2basis(self) -> torch.Tensor:
        return

    @abc.abstractmethod
    def grid_measure(self, n: int) -> torch.Tensor:
        """Returns the domain of the measure discretised on a grid of
        n points.
        
        Parameters
        ----------
        n:
            Number of discretisation points.

        Returns
        -------
        :
            The discretised domain.
        
        """
        return

    @abc.abstractmethod
    def eval_int_basis(self, xs: torch.Tensor) -> torch.Tensor:
        """Computes the indefinite integral of the product of each
        basis function and the weight function at a set of points on 
        the interval [-1, 1].

        Parameters
        ----------
        xs: 
            The set of points at which to evaluate the indefinite 
            integrals of the product of the basis function and the 
            weights.
        
        Returns
        -------
        :
            An array of the results. Each row contains the values of 
            the indefinite integrals for each basis function for a 
            single value of x.

        References
        ----------
        Cui et al. (2023), Appendix A.

        """
        return
        
    @abc.abstractmethod
    def eval_int_basis_newton(
        self, 
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the indefinite integral of the product of each 
        basis function and the weight function, and the product of the
        derivative of this integral with the weight function, at a set 
        of points on the interval [-1, 1]. 
        
        Parameters
        ----------
        xs: 
            The set of points at which to evaluate the indefinite 
            integrals of the product of the basis function and the 
            weights.
        
        Returns
        -------
        :
            An array of the integrals, and an array of the derivatives. 
            Each row contains the values of the indefinite integrals /
            derivatives for each basis function, at a single value of x.
        
        """
        return

    def update_sampling_nodes(
        self, 
        sampling_nodes: torch.Tensor
    ) -> None:
        
        self.sampling_nodes = sampling_nodes 
        self.cdf_basis2node = self.eval_int_basis(self.sampling_nodes)
        return
    
    def eval_int(
        self, 
        coef: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        basis_vals = self.eval_int_basis(x)

        if coef.shape[1] == 1:
            f = (basis_vals @ coef).flatten()
            return f

        if coef.shape[1] != x.numel():
            raise Exception("Dimension mismatch.")

        # TODO: figure out if the flatten call is needed
        # TODO: check dimension of sum
        f = torch.sum(basis_vals * coef.T, 1).flatten()
        return f
    
    def eval_int_search(
        self,
        coef: torch.Tensor, 
        cdf_poly_base,
        rhs: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        f = self.eval_int(coef, x)
        f = f - cdf_poly_base - rhs
        return f

    def eval_int_newton(
        self, 
        coef: torch.Tensor, 
        cdf_poly_base, 
        rhs, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        
        # basis and derivative of basis
        b, db = self.eval_int_basis_newton(x)

        if coef.shape[1] > 1:
            if coef.shape[1] == x.numel():
                f = torch.sum(b * coef.T, 1)
                df =  torch.sum(db * coef.T, 1)
            else:
                raise Exception("Dimension mismatch.")
        else:
            f = b * coef 
            df = b * coef

        f = f - cdf_poly_base - rhs
        return f, df
    
    def eval_cdf(self, pdf: torch.Tensor, r: torch.Tensor):
        """I think pk is the PDF,
        and r is a vector of points at which to evaluate the CDF.
        
        Returns 
        -------
        :
            The values of the CDF evaluated at each point.
        
        """

        self.check_pdf_positive(pdf)

        if pdf.dim() == 1:
            pdf = pdf[:, None]

        if pdf.shape[1] > 1 and pdf.shape[1] != r.numel():
            raise Exception("Dimension mismatch.")
        
        coef = self.node2basis @ pdf
        poly_base = self.cdf_basis2node[0] @ coef
        # Normalising constant
        poly_norm = (self.cdf_basis2node[-1] - self.cdf_basis2node[0]) @ coef

        mask_left = r < self.sampling_nodes[0]
        mask_right = r > self.sampling_nodes[-1]
        mask_inside = ~(mask_left | mask_right)

        zs = torch.zeros_like(r)

        if torch.any(mask_inside):
            if pdf.shape[1] == 1:
                zs[mask_inside] = self.eval_int(coef, r[mask_inside]) - poly_base
            else:
                tmp = self.eval_int(coef[:, mask_inside], r[mask_inside])
                zs[mask_inside] = tmp.flatten() - poly_base[mask_inside].flatten()
        
        if torch.any(mask_right):
            if pdf.shape[1] == 1:
                zs[mask_right] = poly_norm 
            else:
                zs[mask_right] = poly_norm[mask_right]

        zs = zs / poly_norm.flatten()

        # z(isnan(z)) = eps;
        # z(isinf(z)) = 1-eps;
        # z(z>(1-eps)) = 1-eps;
        # z(z<eps) = eps;
    
        return zs

    def eval_int_deriv(
        self, 
        pk: torch.Tensor, 
        r: torch.Tensor
    ) -> torch.Tensor:
        
        coef = self.node2basis @ pk 
        base = self.cdf_basis2node[0] @ coef

        if pk.shape[1] == 1:
            z = self.eval_int(self, coef, r) - base 
        else:
            tmp = self.eval_int(coef, r)
            z = tmp - base
        
        return z
    
    def invert_cdf(
        self, 
        pdf: torch.Tensor, 
        xi: torch.Tensor
    ) -> torch.Tensor:

        self.check_pdf_positive(pdf)

        if pdf.dim() == 1:
            pdf = pdf[:, None]

        # data_size = pdf.shape[0]

        if pdf.shape[1] > 1 and pdf.shape[1] != xi.numel():
            raise Exception("Dimension mismatch.")
        
        coef = self.node2basis @ pdf

        cdf_poly_nodes = self.cdf_basis2node @ coef
        cdf_poly_base = cdf_poly_nodes[0]
        cdf_poly_nodes = cdf_poly_nodes - cdf_poly_base
        cdf_poly_norm = cdf_poly_nodes[-1]

        r = torch.zeros_like(xi)

        if pdf.shape[0] == 1:
            rhs = xi * cdf_poly_norm 
            ind = torch.sum(torch.reshape(cdf_poly_nodes, 1, -1) < rhs, 1) # TODO: check this...
        else:
            rhs = xi * cdf_poly_norm  # vertical??
            ind = torch.sum(cdf_poly_nodes < rhs, 1)  # check
        
        mask_1 = (ind == 0 | xi < 1e-8)  # TODO: fix eps (below also)
        mask_3 = (ind == self.sampling_nodes.numel() | xi >= 1.0 - 1e-8)
        mask_2 = not (mask_1 | mask_3)

        if torch.any(mask_1):
            r[mask_1] = self.sampling_nodes[0]

        if torch.any(mask_3):
            r[mask_3] = self.sampling_nodes[-1]
        
        if torch.any(mask_2):
            a = self.sampling_nodes[int(ind[mask_2])]
            b = self.sampling_nodes[int(ind[mask_2]+1)]
            if pdf.shape[0] == 1:
                r[mask_2] = self.newton(coef, cdf_poly_base, rhs[mask_2], a, b)
            else:
                r[mask_2] = self.newton(coef[mask_2], cdf_poly_base[mask_2], rhs[mask_2], a, b)
        
        # TODO: isnan stuff??
        return r

    def newton(
        self,
        coef: torch.Tensor, 
        cdf_poly_base: torch.Tensor, 
        rhs: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        
        fa = self.eval_int_search(self, coef, cdf_poly_base, rhs, a)
        fb = self.eval_int_search(self, coef, cdf_poly_base, rhs, b)

        raise NotImplementedError()
        

    """
    function c = newton(obj, coef, cdf_poly_base, rhs, a, b)
            %i = 0;
            fa = eval_int_search(obj,coef,cdf_poly_base,rhs,a);
            fb = eval_int_search(obj,coef,cdf_poly_base,rhs,b);
            if any(fb.*fa > 0)
                disp(['Root finding: initial guesses on one side, # violation: ' num2str(sum(fb.*fa > 0))])
                %disp([a(fb.*fa > 0), b(fb.*fa > 0)])
            end
            c = b - fb.*(b - a)./(fb - fa);  % Regula Falsi
            rf_flag = true;
            for iter = 1:obj.num_Newton
                cold = c;
                [f,df] = eval_int_newton(obj, coef, cdf_poly_base, rhs, cold);
                step = f./df;
                step(isnan(step)) = 0;
                c = cold - step;
                I1 = c<a;
                I2 = c>b;
                I3 = ~I1 & ~I2;
                c  = a.*I1 + b.*I2 + c.*I3;
                if ( norm(f, Inf) < obj.tol ) || ( norm(step, Inf) < obj.tol )
                    rf_flag = false;
                    break;
                end
            end
            %norm(f, Inf)
            if rf_flag
                disp('newton does not converge, continue with regula falsi')
                fc = eval_int_search(obj,coef,cdf_poly_base,rhs,c);
                I1 = (fc < 0);
                I2 = (fc > 0);
                I3 = ~I1 & ~I2;
                a  = I1.*c + I2.*a + I3.*a;
                b  = I1.*b + I2.*c + I3.*b;
                c = regula_falsi(obj, coef, cdf_poly_base, rhs, a, b);
            end
        end
    end"""


"""
classdef SpectralCDF < OnedCDF
    % SpectralCDF class
    %
    % For Fourier basis, FourierCDF is used. For other spectral polynomials
    % in bounded domains, we first transform the polynomial to the 2nd
    % Chebyshev basis, and then apply the inversion. See Chebyshev2ndCDF.
    %
    % Before applying root findings, a grid search based on sampling_nodes
    % is applied to locate the left and right boundary of root finding.
    %
    % See also ChebyshevCDF and FourierCDF.
    
    properties
        sampling_nodes(1,:)
        cdf_basis2node(:,:)
    end

    methods (Abstract)
        grid_measure(obj)
        eval_int_basis(obj)
        eval_int_basis_newton(obj)
    end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function r = invert_cdf(obj, pk, xi)
            if (sum(pk(:)<0)>0)
                disp(['negative pdf ' num2str(sum(pk(:)<0))])
            end
            data_size = size(pk,2);
            coef = obj.node2basis*pk;
            cdf_poly_nodes = obj.cdf_basis2node*coef;
            %
            cdf_poly_base  = cdf_poly_nodes(1,:);
            cdf_poly_nodes = cdf_poly_nodes - cdf_poly_base;
            cdf_poly_norm  = cdf_poly_nodes(end,:);
            if data_size > 1 && data_size ~= length(xi)
                error('Error: dimenion mismatch')
            end
            %
            xi = reshape(xi,[],1);
            r = zeros(size(xi));
            
            if data_size == 1
                rhs = xi.*cdf_poly_norm; % vertical
                ind = sum(reshape(cdf_poly_nodes,1,[]) < rhs(:),2)';
            else
                rhs = xi(:).*cdf_poly_norm(:); % vertical
                ind = sum(cdf_poly_nodes < reshape(rhs,1,[]), 1);
            end
            mask1 = ind==0 | reshape(xi,1,[])<=eps;
            mask3 = ind==length(obj.sampling_nodes) | reshape(xi,1,[])>=1-eps;
            mask2 = ~(mask1|mask3);
            %
            % left and right tails
            if sum(mask1) > 0
                r(mask1) = obj.sampling_nodes(1);
            end
            if sum(mask3) > 0
                r(mask3) = obj.sampling_nodes(end);
            end
            %
            if sum(mask2) > 0
                a = obj.sampling_nodes(ind(mask2));
                b = obj.sampling_nodes(ind(mask2)+1);
                %
                if data_size == 1
                    r(mask2) = newton(obj, coef, cdf_poly_base, rhs(mask2), a(:), b(:));
                else
                    r(mask2) = newton(obj, coef(:,mask2), reshape(cdf_poly_base(mask2),[],1), rhs(mask2), a(:), b(:));
                end
            end
            %
            r = reshape(r, size(xi));
            r(isnan(r)) = 0.5*(obj.domain(1) + obj.domain(2));
            r(isinf(r)) = 0.5*(obj.domain(1) + obj.domain(2));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function c = regula_falsi(obj, coef, cdf_poly_base, rhs, a, b)
            fa = eval_int_search(obj,coef,cdf_poly_base,rhs,a);
            fb = eval_int_search(obj,coef,cdf_poly_base,rhs,b);
            if any((fb.*fa) > 0)
                disp(['Root finding: initial guesses on one side, # violation: ' num2str(sum((fb.*fa) > 0))])
                %disp([a(fb.*fa > 0), b(fb.*fa > 0)])
            end
            c = b - fb.*(b - a)./(fb - fa);  % Regula Falsi
            cold = inf;
            %i = 2;
            while ( norm(c-cold, Inf) > obj.tol )
                cold = c;
                fc  = eval_int_search(obj,coef,cdf_poly_base,rhs,c);
                if norm(fc, Inf) < obj.tol
                    break;
                end
                I1  = (fc < 0);
                I2  = (fc > 0);
                I3  = ~I1 & ~I2;
                a   = I1.*c + I2.*a + I3.*c;
                b   = I1.*b + I2.*c + I3.*c;
                fa  = I1.*fc + I2.*fa + I3.*fc;
                fb  = I1.*fb + I2.*fc + I3.*fc;
                step    = -fb.*(b - a)./(fb - fa);
                step(isnan(step)) = 0;
                c = b + step;
                %norm(fc, inf)
                %i = i+1;
                %disp(i)
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function c = newton(obj, coef, cdf_poly_base, rhs, a, b)
            %i = 0;
            fa = eval_int_search(obj,coef,cdf_poly_base,rhs,a);
            fb = eval_int_search(obj,coef,cdf_poly_base,rhs,b);
            if any(fb.*fa > 0)
                disp(['Root finding: initial guesses on one side, # violation: ' num2str(sum(fb.*fa > 0))])
                %disp([a(fb.*fa > 0), b(fb.*fa > 0)])
            end
            c = b - fb.*(b - a)./(fb - fa);  % Regula Falsi
            rf_flag = true;
            for iter = 1:obj.num_Newton
                cold = c;
                [f,df] = eval_int_newton(obj, coef, cdf_poly_base, rhs, cold);
                step = f./df;
                step(isnan(step)) = 0;
                c = cold - step;
                I1 = c<a;
                I2 = c>b;
                I3 = ~I1 & ~I2;
                c  = a.*I1 + b.*I2 + c.*I3;
                if ( norm(f, Inf) < obj.tol ) || ( norm(step, Inf) < obj.tol )
                    rf_flag = false;
                    break;
                end
            end
            %norm(f, Inf)
            if rf_flag
                disp('newton does not converge, continue with regula falsi')
                fc = eval_int_search(obj,coef,cdf_poly_base,rhs,c);
                I1 = (fc < 0);
                I2 = (fc > 0);
                I3 = ~I1 & ~I2;
                a  = I1.*c + I2.*a + I3.*a;
                b  = I1.*b + I2.*c + I3.*b;
                c = regula_falsi(obj, coef, cdf_poly_base, rhs, a, b);
            end
        end
    end
    
end
"""