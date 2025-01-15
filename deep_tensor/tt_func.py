from typing import Callable, Tuple
import warnings

import torch

from .approx_bases import ApproxBases
from .directions import Direction
from .input_data import InputData
from .options import TTOptions
from .polynomials import Basis1D, Piecewise, Spectral
from .tools import deim, maxvol, reshape_matlab
from .tt_data import TTData
from .utils import als_info

MAX_COND = 1.0e+5


class TTFunc():

    def __init__(
        self, 
        target_func: Callable[[torch.Tensor], torch.Tensor], 
        bases: ApproxBases, 
        options: TTOptions, 
        input_data: InputData,
        tt_data: TTData|None
    ):
        """A functional tensor-train approximation for a function 
        mapping from some subset of R^d to some subset of R.

        Parameters
        ----------
        target_func:
            Maps an n * d matrix containing samples from the local 
            domain ([-1, 1]^d) to an n-dimensional vector containing 
            the values of the target function at each sample.
        bases:
            The bases associated with the approximation domain.
        options:
            Options used when constructing the FTT approximation to
            the target function.
        input_data:
            Data used for initialising and evaluating the quality of 
            the FTT approximation to the target function.
        tt_data:
            Data used to construct the FTT approximation to the target
            function.

        """
        
        self.bases = bases 
        self.dim = bases.dim
        self.options = options
        self.input_data = input_data
        self.data = TTData() if tt_data is None else tt_data

        # if isinstance(arg, ApproxFunc):
        #     self.options = arg.options
        
        self.input_data.set_debug(target_func, self.bases)
        self.num_eval = 0
        self.errors = torch.zeros(self.bases.dim)
        self.l2_err = torch.inf
        self.linf_err = torch.inf

        self.input_data.set_samples(self.bases, self.sample_size)
        self.cross(target_func)
        return
        
    @property 
    def rank(self) -> torch.Tensor:
        """The ranks of each tensor core."""
        return self.data.rank

    @property
    def use_amen(self) -> bool:
        """Whether to use AMEN."""
        return self.options.tt_method.lower() == "amen"
        
    @property
    def sample_size(self):
        """An upper bound on the total number of samples required to 
        construct a FTT approximation to the target function.
        """
        n = (self.options.init_rank 
             + self.options.kick_rank * (1+self.options.max_als))
        n *= self.dim
        return n

    def _print_info_header(self) -> None:

        info_headers = [
            "ALS", 
            "Max Local Error", 
            "Mean Local Error", 
            "Max Rank", 
            "Func Evals"
        ]
        
        if self.input_data.is_debug:
            info_headers += [
                "Max Debug Error", 
                "Mean Debug Error"
            ]

        als_info(" | ".join(info_headers))
        return

    def _print_info(
        self,
        als_iter: int,
        indices: torch.Tensor
    ) -> None:
        """Prints some diagnostic information about the current ALS 
        iteration.
        """

        diagnostics = [
            f"{als_iter:=3}", 
            f"{torch.max(self.errors[indices]):=15.5e}",
            f"{torch.mean(self.errors[indices]):=16.5e}",
            f"{torch.max(self.rank):=8}",
            f"{self.num_eval:=10}"
        ]

        if self.input_data.is_debug:
            diagnostics += [
                f"{self.linf_err:=15.5e}",
                f"{self.l2_err:=16.5e}"
            ]

        als_info(" | ".join(diagnostics))
        return

    def _compute_cross_iter_fixed_rank(
        self, 
        target_func: Callable,
        indices: torch.Tensor
    ) -> None:

        for k in indices:
                    
            x_left = self.data.interp_x[int(k-1)]
            x_right = self.data.interp_x[int(k+1)]
            
            F_k = self.build_block_local(target_func, x_left, x_right, k) 
            self.errors[k] = self.get_error_local(F_k, k)
            self.build_basis_svd(F_k, k)

        return
    
    def _compute_cross_iter_random(
        self,
        target_func: Callable[[torch.Tensor], torch.Tensor],
        indices: torch.Tensor
    ) -> None:
        
        for k in indices:
            
            x_left = self.data.interp_x[int(k-1)].clone()
            x_right = self.data.interp_x[int(k+1)].clone()
            enrich = self.input_data.get_samples(self.options.kick_rank)

            F_k = self.build_block_local(target_func, x_left, x_right, k)
            self.errors[k] = self.get_error_local(F_k, k)

            if self.data.direction == Direction.FORWARD:
                F_enrich = self.build_block_local(target_func, x_left, enrich[:, k+1:], k)
                F_full = torch.concatenate((F_k, F_enrich), dim=2)
            else:
                F_enrich = self.build_block_local(target_func, enrich[:, :k], x_right, k)
                F_full = torch.concatenate((F_k, F_enrich), dim=0)

            self.build_basis_svd(F_full, k)

        return None
    
    def _compute_cross_iter_amen(
        self, 
        func: Callable, 
        indices: torch.Tensor
    ):
        
        for k in indices:
            
            if self.data.direction == Direction.FORWARD:

                x_left = self.data.interp_x[int(k-1)]
                x_right = self.data.interp_x[int(k+1)]
                r_left = self.data.res_x[int(k-1)] # TODO: add the left-most and right-most versions of these to make this well-defined 
                r_right = self.data.res_x[int(k+1)]
                w_left = self.data.res_w[int(k-1)]

                # Evaluate the interpolant function at x_k nodes
                F = self.build_block_local(func, x_left, x_right, k)
                self.errors[k] = self.get_error_local(F, k)

                # Evaluate residual function at x_k nodes
                F_res = self.build_block_local(func, r_left, r_right, k)

                # Evaluate update function at x_k nodes
                if k > 0:
                    F_up = self.build_block_local(func, x_left, r_right, k)
                else:
                    F_up = F_res

                raise NotImplementedError("TODO: finish")

            else:
                raise NotImplementedError()

        return 

    def _compute_final_block(self, func: Callable) -> None:

        if self.data.direction == Direction.FORWARD:
            k = self.dim-1 
        else:
            k = 0

        x_left = self.data.interp_x[int(k-1)]
        x_right = self.data.interp_x[int(k+1)]
        self.data.cores[k] = self.build_block_local(func, x_left, x_right, k)
        return

    def _select_points_piecewise(
        self,
        H: torch.Tensor,
        poly: Piecewise
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.options.int_method == "qdeim":
            raise NotImplementedError()
            # indices, core = qdeim(H)
            # interp_atx = H[indices]

        elif self.options.int_method == "deim":
            indices, A = deim(H)
            interp_atx = H[indices]

        elif self.options.int_method == "maxvol":
            indices, A = maxvol(H)
            interp_atx = H[indices]
        
        if (cond := torch.linalg.cond(interp_atx)) > MAX_COND:
            msg = f"Poor condition number in interpolation: {cond}."
            warnings.warn(msg)

        return indices, A, interp_atx
    
    def _select_points_spectral(
        self,
        H: torch.Tensor,
        poly: Spectral
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # TODO: this is probably available somewhere else.
        rank_prev = torch.tensor(H.shape[0] / poly.cardinality)
        rank_prev = rank_prev.round().int()
        
        nodes = poly.basis2node @ reshape_matlab(H, (poly.cardinality, -1))
        nodes = reshape_matlab(nodes, (poly.cardinality * rank_prev, -1))

        if self.options.int_method == "qdeim":
            raise NotImplementedError()
        
        elif self.options.int_method == "deim":
            msg = "DEIM is not supported for spectral polynomials."
            raise Exception(msg)

        elif self.options.int_method == "maxvol":
            indices, _ = maxvol(nodes)
            interp_atx = nodes[indices]
            core = H @ torch.linalg.inv(interp_atx)
        
        if (cond := torch.linalg.cond(interp_atx)) > 1e+5:
            msg = f"Poor condition number in interpolation ({cond})."
            warnings.warn(msg)

        return indices, core, interp_atx

    def compute_relative_error(self) -> None:
        """TODO: write docstring."""

        if not self.input_data.is_debug:
            return
        
        ps_approx = self.eval_local(self.input_data.ls_debug)
        self.l2_err, self.linf_err = self.input_data.relative_error(ps_approx)
        return

    def initialise_cores(self) -> None:
        """Initialises the cores and interpolation points in each 
        dimension.
        """

        for k in range(self.dim):

            core_shape = [
                1 if k == 0 else self.options.init_rank, 
                self.bases.polys[k].cardinality,
                self.options.init_rank if k != self.dim-1 else 1
            ]

            self.data.cores[k] = torch.rand(core_shape)

            samples = self.input_data.get_samples(self.options.init_rank)
            self.data.interp_x[k] = samples[:, k:]

        self.data.interp_x[-1] = torch.tensor([])
        self.data.interp_x[self.dim] = torch.tensor([])

        return

    def initialise_amen(self) -> None:
        """TODO: figure out how AMEN works and tidy this up."""

        if self.data.res_x == {}:
            # Define set of nested interpolation points for the residual
            for k in range(self.dim-1, -1, -1):
                samples = self.input_data.get_samples(self.options.kick_rank)
                if self.data.direction == Direction.FORWARD:
                    self.data.res_x[k] = samples[:, k:]
                else:
                    self.data.res_x[k] = samples[:, :(k+1)]

        # TODO: tidy up the below
        if self.data.res_w == {}:
            if self.data.direction == Direction.FORWARD:
                self.data.res_w[0] = torch.ones((self.options.kick_rank, self.data.cores[k].shape[-1]))
                for k in range(1, self.dim):
                    res_shape = (self.data.cores[k].shape[0], self.options.kick_rank)
                    self.data.res_w[k] = torch.ones(res_shape)

            else:
                for k in range(self.dim-1):
                    res_shape = (self.options.kick_rank, self.data.cores[k].shape[-1])
                    self.data.res_w[k] = torch.ones(res_shape)
                self.data.res_x[self.dim-1] = torch.ones((self.data.cores[k].shape[0], self.options.kick_rank))
        
        return

    def build_block_local(
        self, 
        target_func: Callable[[torch.Tensor], torch.Tensor], 
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, int]:
        """Evaluates the function being approximated at a (reduced) set 
        of interpolation points, and returns the corresponding
        local coefficient matrix.

        Parameters
        ----------
        target_func: 
            The function being approximated.
        x_left:
            An r_{k-1} * {k-1} matrix containing a set of interpolation
            points for dimensions 1, ..., {k-1}.
        x_right:
            An r_{k+1} * {k+1} matix containing a set of interpolation 
            points for dimensions {k+1}, ..., d.
        k:
            The dimension in which interpolation is being carried out.

        Returns
        -------
        F_k: 
            An r_{k-1} * n_{k} * r_{k} tensor containing the values of 
            the function evaluated at each interpolation point.

        References
        ----------
        Cui and Dolgov (2022). Deep composition of tensor-trains using 
        squared inverse Rosenblatt transports.
        
        """

        num_left = 1 if x_left.numel() == 0 else x_left.shape[0]
        num_right = 1 if x_right.numel() == 0 else x_right.shape[0]

        poly = self.bases.polys[k]
        nodes = poly.nodes[:, None]

        # Form the Cartesian product of the index sets and the nodes
        # corresponding to the basis of the current dimension
        if x_left.numel() == 0:

            params = torch.hstack((
                nodes.repeat_interleave(num_right, dim=0),
                x_right.repeat(poly.cardinality, 1)
            ))

        elif x_right.numel() == 0:

            params = torch.hstack((
                x_left.repeat_interleave(poly.cardinality, dim=0),
                nodes.repeat(num_left, 1)
            ))

        else:

            params = torch.hstack((
                x_left.repeat_interleave(poly.cardinality * num_right, dim=0),
                nodes.repeat_interleave(num_right, dim=0).repeat(num_left, 1),
                x_right.repeat(num_left * poly.cardinality, 1)
            ))
        
        F_k = target_func(params).reshape(num_left, poly.cardinality, num_right)

        # TODO: could be a separate method eventually
        if isinstance(poly, Spectral): 
            F_k = F_k.permute(2, 0, 1).reshape(num_left * num_right, -1).T
            F_k = poly.node2basis @ F_k
            F_k = F_k.T.reshape(num_right, num_left, -1).permute(1, 2, 0)

        self.num_eval += params.shape[0]
        return F_k

    def truncate_local(
        self,
        F: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Truncates the SVD for a given TT block, F.

        Parameters
        ----------
        F:
            An r_{k_next} * (n_{k} * r_{k_prev}) matrix containing 
            the coefficients associated with the kth tensor core.
        k:
            The index of the current dimension.
        
        Returns
        -------
        B:
            Matrix containing the left singular vectors of F after 
            truncation.
        A: 
            Matrix containing the transpose of the product of the 
            singular values and the right-hand singular vectors after
            truncation. 
        rank:
            The number of singular values of F that were retained.

        """

        poly = self.bases.polys[k]
        if isinstance(poly, Piecewise):
            nr_prev = F.shape[0]
            F = poly.mass_R @ F.T.reshape(-1, poly.cardinality).T
            F = F.T.reshape(-1, nr_prev).T
            
        U, s, Vh = torch.linalg.svd(F, full_matrices=False)
            
        energies = torch.flip(s**2, dims=(0,)).cumsum(dim=0)
        tol = 0.1 * energies[-1] * self.options.local_tol ** 2
        
        rank = torch.sum(energies > tol)
        rank = torch.clamp(rank, 1, self.options.max_rank)

        B = U[:, :rank]
        A = (s[:rank] * Vh[:rank].T).T

        if isinstance(poly, Piecewise):
            B = B.T.reshape(-1, poly.cardinality).T
            B = torch.linalg.solve(poly.mass_R, B)
            B = B.T.reshape(-1, nr_prev).T
 
        return B, A, rank

    def build_basis_svd(
        self, 
        F_k: torch.Tensor, 
        k: torch.Tensor|int
    ) -> None:
        """TODO: write docstring...
        
        Parameters
        ----------
        F_k:
            An r_{k-1} * n_{k} * r_{k} tensor containing the 
            coefficients of the kth TT block.
        k:
            The index of the dimension corresponding to the basis 
            being constructed.

        Returns
        -------
        None
            
        """

        k = int(k)
        k_prev = int(k - self.data.direction.value)
        k_next = int(k + self.data.direction.value)
        
        poly = self.bases.polys[k]
        interp_x_prev = self.data.interp_x[k_prev]
        core_next = self.data.cores[k_next]

        n_left, n_nodes, n_right = F_k.shape 
        r_0_next, n_nodes_next, r_1_next = core_next.shape

        if self.data.direction == Direction.FORWARD:
            F_k = F_k.permute(2, 0, 1).reshape(n_right, n_left * n_nodes).T
            r_prev = n_left
        else: 
            F_k = F_k.permute(0, 2, 1).reshape(n_left, n_nodes * n_right).T
            r_prev = n_right

        B, A, rank = self.truncate_local(F_k, k)

        indices, core, interp_atx = self.select_points(B, k)

        couple = reshape_matlab(interp_atx @ A, (rank, -1))
        interp_x = self.get_local_index(poly, interp_x_prev, indices)

        if self.data.direction == Direction.FORWARD:
            
            core = core.T.reshape(rank, r_prev, n_nodes).permute(1, 2, 0)

            couple = couple[:, :r_0_next].T.reshape(r_0_next, -1).T
            
            core_next = couple @ core_next.permute(2, 1, 0).reshape(-1, r_0_next).T
            core_next = reshape_matlab(core_next, (rank, n_nodes_next, r_1_next))

        else:
            
            core = reshape_matlab(core, (n_nodes, r_prev, rank))
            core = core.permute(2, 0, 1)

            couple = couple[:, :r_1_next].T
            couple = reshape_matlab(couple, (r_1_next, -1))

            core_next = reshape_matlab(core_next, (-1, r_1_next))
            core_next = core_next @ couple 
            core_next = reshape_matlab(core_next, (r_0_next, n_nodes_next, rank))

        self.data.cores[k] = core
        self.data.interp_x[k] = interp_x 
        self.data.cores[k_next] = core_next

        return
    
    def build_basis_amen(
        self, 
        F: torch.Tensor,
        F_res: torch.Tensor,
        F_up: torch.Tensor,
        k: torch.Tensor|int
    ) -> None:
        """TODO: finish"""
        
        k = int(k)
        k_prev = int(k - self.data.direction.value)
        k_next = int(k + self.data.direction.value)
        
        poly = self.bases.polys[k]
        interp_x_prev = self.data.interp_x[k_prev]
        core_next = self.data.cores[k_next]

        num_b_left, num_nodes, num_b_right = F.shape
        num_r_left, _, num_r_right = F_res.shape
        rank_0_next, num_nodes_next, rank_1_next = core_next.shape

        if self.data.direction == Direction.FORWARD:
            F = F.permute(1, 0, 2)
            F = reshape_matlab(F, (num_b_left * num_nodes, num_b_right))
            F_up = F_up.permute(1, 0, 2)
            F_up = reshape_matlab(F_up, (num_b_left * num_nodes, num_b_right))
            rank_prev = num_b_left 
            rank_next = rank_0_next

        else:
            F = F.permute(1, 2, 0)
            F = reshape_matlab(F, (num_nodes * num_b_right, num_b_left))
            F_up = F_up.permute(1, 2, 0)
            F_up = reshape_matlab(F_up, (num_nodes * num_b_right, num_b_left))
            rank_prev = num_b_right 
            rank_next = rank_1_next

        B, A, rank = self.truncate_local(F, k)

        if self.data.direction == Direction.FORWARD:

            raise NotImplementedError()
            # temp_r = reshape_matlab(A, (rank, rank_0_next)) @ res_w_r
            # temp_r = reshape_matlab(temp_r, (-1, ))

        return

    def get_error_local(
        self,
        F: torch.Tensor,
        k: torch.Tensor
    ) -> torch.Tensor:
        """Returns the ..."""
        core = self.data.cores[int(k)]
        return (core-F).abs().max() / F.abs().max()

    def get_local_index(
        self,
        poly: Basis1D, 
        interp_x_prev: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Builds the local nested index set.
        
        TODO: write docstring for this.
        """

        if interp_x_prev.numel() == 0:
            interp_x = poly.nodes[indices][:, None]
            return interp_x

        rank_prev = interp_x_prev.shape[0]

        # Form the full Cartesian product of something...
        i_pair = torch.vstack((
            torch.arange(rank_prev).repeat_interleave(poly.cardinality),
            torch.arange(poly.cardinality).repeat(rank_prev)
        ))
        i_select = i_pair[:, indices]

        if self.data.direction == Direction.FORWARD:
            interp_x = torch.hstack((
                interp_x_prev[i_select[0]],
                poly.nodes[i_select[1]][:, None]
            ))
        else:
            interp_x = torch.hstack((
                poly.nodes[i_select[1]][:, None],
                interp_x_prev[i_select[0]]
            ))

        return interp_x

    def eval(self, xs: torch.Tensor) -> torch.Tensor:
        """Evaluates the approximated function at a set of points in 
        the approximation domain.
        
        Parameters
        ----------
        xs:
            A matrix containing n sets of d-dimensional input 
            variables in the approximation domain. Each row contains a
            single input variable.
            
        Returns
        -------
        fxs:
            An n-dimensional vector containing the values of the 
            function at each x value.
        """
        ls = self.bases.approx2local(xs)[0]
        fxs = self.eval_local(ls)
        return fxs

    def eval_local(
        self, 
        ls: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the approximation to the target function for a set 
        of samples in the local domain ([-1, 1]^d).

        Parameters
        ----------
        ls:
            An n * d matrix containing samples in the local domain.
        
        Returns
        -------
        ps:
            An n-dimensional vector containing the result of evaluating
            the target function at each element in ls. 
            
        """
        ps = self.eval_block(ls, self.data.direction)
        return ps
    
    def eval_block(
        self, 
        ls: torch.Tensor,
        direction: Direction
    ) -> torch.Tensor:
        """Evaluates the functional tensor train approximation to the 
        target function for either the first or last k variables, for a 
        set of points in the local domain ([-1, 1]).
        
        Parameters
        ----------
        ls:
            A set of points in the local domain.
        
        Returns
        -------
        fls:
            The value of the FTT approximation to the target function
            at each point in ls.

        TODO: figure out exactly what's going on in here (I think 
        ultimately it just takes the product of all the cores and basis 
        functions).
            
        """

        num_l, dim_l = ls.shape

        fls = torch.ones((num_l, 1))

        if direction == Direction.FORWARD:

            for k in range(min(dim_l, self.dim)):

                rank_p, num_nodes, rank_k = self.data.cores[k].shape

                A_k = (self.data.cores[k]
                       .permute(2, 0, 1)
                       .reshape(rank_p * rank_k, num_nodes).T)

                G_k = (self.bases.polys[k].eval_radon(A_k, ls[:, k]).T
                       .reshape(rank_k, rank_p, num_l)
                       .swapdims(2, 1)
                       .reshape(rank_k, num_l * rank_p).T)
                
                ii = torch.arange(num_l).repeat(rank_p)
                jj = (torch.arange(rank_p * num_l)
                      .reshape(num_l, rank_p).T
                      .flatten())
                indices = torch.vstack((ii[None, :], jj[None, :]))
                size = (num_l, rank_p * num_l)
                B = torch.sparse_coo_tensor(indices, fls.T.flatten(), size)

                fls = B @ G_k

        else:
            
            x_inds = torch.arange(dim_l-1, -1, -1)
            t_inds = torch.arange(self.bases.dim-1, -1, -1)
            
            for i in range(min(dim_l, self.bases.dim)):
                
                j = int(t_inds[i])
                
                rank_p, num_nodes, rank_j = self.data.cores[j].shape

                A_k = self.data.cores[j].permute(1, 2, 0)
                A_k = reshape_matlab(A_k, (num_nodes, -1))

                G_k = self.bases.polys[j].eval_radon(A_k, ls[:, x_inds[i]])
                G_k = reshape_matlab(G_k, (num_l, rank_j, rank_p))
                G_k = G_k.permute(1, 0, 2)
                G_k = reshape_matlab(G_k, (rank_j * num_l, rank_p))

                ii = torch.arange(num_l * rank_j)
                jj = torch.arange(num_l).repeat_interleave(rank_j)
                
                indices = torch.vstack((ii[None, :], jj[None, :]))
                size = (rank_j * num_l, num_l)

                B = torch.sparse_coo_tensor(indices, fls.T.flatten(), size)
                fls = G_k.T @ B

        return fls.squeeze()
    
    def grad_reference(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the gradient of the approximation to the target 
        function for a set of reference variables.
        """
        raise NotImplementedError()

    def grad(self, xs: torch.Tensor) -> torch.Tensor:
        """Evaluates the gradient of the approximation to the target 
        function at a set of points in the approximation domain.
        
        Parameters
        ----------
        xs: 
            A matrix containing n sets of d-dimensional input 
            variables in the approximation domain. Each row contains a
            single input variable.

        Returns
        -------
        gxs:
            TODO: finish this once grad_reference is done.

        """
        zs, dzdxs = self.bases.approx2local(xs)
        gzs, fxs = self.grad_reference(self, zs)
        gxs = gzs * dzdxs
        return gxs, fxs
    
    def int_reference(self):
        """Integrates the approximation to the target function over the 
        reference domain (TODO: check this).
        """
        raise NotImplementedError()

    def select_points(
        self,
        H: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds the cross indices.

        Parameters
        ----------
        H:
            TODO: write this
        k: 
            The index of the current dimension.
        
        Returns
        -------
        indices:
            The set of indices of the (approximate) maximum volume 
            submatrix of H.
        core:
            The corresponding (unfolded) tensor core.
        interp_atx:
            The nodes of the basis of the current dimension 
            corresponding to the set of indices of the maximum volume
            submatrix.
        
        """

        poly = self.bases.polys[k]

        if isinstance(poly, Piecewise):
            return self._select_points_piecewise(H, poly)
        elif isinstance(poly, Spectral):
            return self._select_points_spectral(H, poly)
    
        raise Exception("Unknown polynomial encountered.")

    def is_finished(
        self, 
        als_iter: int,
        indices: torch.Tensor
    ) -> bool:
        """Returns True if the maximum number of ALS iterations has 
        been reached or the desired error tolerance is met, and False 
        otherwise.
        """
        
        max_iters = als_iter == self.options.max_als
        max_error_tol = torch.max(self.errors[indices]) < self.options.als_tol
        l2_error_tol = self.l2_err < self.options.als_tol # TODO: check where l2_err actually gets updated.

        return max_iters or max_error_tol or l2_error_tol

    def cross(
        self, 
        target_func: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """Cross iterations for building the tensor train.

        Parameters
        ----------
        func: 
            Returns the target function evaluated at a given point in 
            the reference domain.
        
        Returns 
        -------
        None
        
        """

        self._print_info_header()
        als_iter = 0

        if self.data.cores == {}:
            self.data.direction = Direction.FORWARD 
            self.initialise_cores()
        else:
            # Prepare for the next iteration
            self.data.reverse_direction()
        
        if self.use_amen:
            self.initialise_amen()

        while True:

            if self.data.direction == Direction.FORWARD:
                indices = torch.arange(self.dim-1)
            else:
                indices = torch.arange(self.dim-1, 0, -1)
            
            if self.options.tt_method == "fixed_rank":
                self._compute_cross_iter_fixed_rank(target_func, indices)
            
            elif self.options.tt_method == "random":
                self._compute_cross_iter_random(target_func, indices)

            elif self.options.tt_method == "amen":
                self._compute_cross_iter_amen(target_func, indices)  # TODO: implement this. 
            
            else: 
                raise Exception("Unknown TT method.")

            als_iter += 1

            if self.is_finished(als_iter, indices): 
                self._compute_final_block(target_func)

            self.compute_relative_error()

            if self.is_finished(als_iter, indices):
                self._print_info(als_iter, indices)
                als_info(f"ALS complete.")
                als_info(f"Final TT ranks: {[int(r) for r in self.rank]}.")
                return

            else:
                self._print_info(als_iter, indices)
                self.data.reverse_direction()