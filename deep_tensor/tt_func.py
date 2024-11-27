from typing import Callable, Tuple
import warnings

import torch

from .approx_bases import ApproxBases
from .approx_func import ApproxFunc
from .input_data import InputData
from .options import TTOptions
from .polynomials import Basis1D, Piecewise, Spectral
from .directions import Direction
from .tools import maxvol, reshape_matlab
from .utils import info


class TTFunc(ApproxFunc):

    def __init__(
        self, 
        func: Callable, 
        bases: ApproxBases, 
        options: TTOptions, 
        input_data: InputData
    ):
        """A functional tensor-train approximation for a function 
        mapping from $\mathbb{R}^{d}$ to $\mathbb{R}$.

        Parameters
        ----------
        func:
            A function ($\mathbb{R}^{d} \rightarrow \mathbb{R}$) that 
            takes as input a $n \times d$ matrix and returns an 
            $n$-dimensional vector.
        
        TODO: finish

        """
        
        super().__init__(func, bases, options, input_data)
        self.input_data.set_samples(self.bases, self.sample_size)
        self.cross(func)
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
        n *= self.bases.dim
        return n

    def initialise_cores(self) -> None:
        """Initialises the cores and interpolation points in each 
        dimension.
        """

        for k in range(self.bases.dim):

            core_shape = [
                1 if k == 0 else self.options.init_rank, 
                self.bases.polys[k].cardinality,
                self.options.init_rank if k != self.bases.dim-1 else 1
            ]

            self.data.cores[k] = torch.rand(core_shape)

            samples = self.input_data.get_samples(self.options.init_rank)
            self.data.interp_x[k] = samples[:, k:]

        self.data.interp_x[-1] = torch.tensor([])
        self.data.interp_x[self.bases.dim] = torch.tensor([])

        return

    def build_block_local(
        self, 
        func: Callable, 
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, int]:
        """Evaluates the function being approximated at a (reduced) set 
        of interpolation points, and returns the corresponding
        local coefficient matrix.

        Parameters
        ----------
        func: 
            The function being approximated.
        k:
            The dimension in which interpolation is being carried out.

        Returns
        -------
        f: 
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
                nodes.repeat(num_right, 1),
                x_right.repeat_interleave(poly.cardinality, dim=0)
            ))

        elif x_right.numel() == 0:

            params = torch.hstack((
                x_left.repeat(poly.cardinality, 1),
                nodes.repeat_interleave(num_left, dim=0)
            ))

        else:

            params = torch.hstack((
                x_left.repeat(poly.cardinality*num_right, 1),
                nodes.repeat_interleave(num_left, dim=0).repeat(num_right, 1),
                x_right.repeat_interleave(num_left * poly.cardinality, dim=0)
            ))
        
        f = func(params)
        f = reshape_matlab(f, (num_left, poly.cardinality, num_right))

        if isinstance(poly, Spectral):
            f = f.permute(1, 0, 2)
            f = poly.node2basis @ reshape_matlab(f, (poly.cardinality, -1))
            f = reshape_matlab(f, (poly.cardinality, num_left, num_right))
            f = f.permute(1, 0, 2)

        self.num_eval += params.shape[0]
        self.errors[k] = self.get_error_local(f, k)

        return f

    def truncate_local(
        self,
        F: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Truncates the SVD for a given TT block, F.

        Parameters
        ----------
        F:
            The TT block. Has dimensions TODO
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

        poly = self.bases.polys[int(k)]

        if isinstance(poly, Piecewise):
            # TODO: figure out what nn is
            nn = F.shape[0]
            F = poly.mass_R @ reshape_matlab(F, (poly.cardinality, -1))
            F = reshape_matlab(F, (nn, -1))
            
        U, s, Vh = torch.linalg.svd(F, full_matrices=False)
            
        energies = torch.cumsum(torch.flip(s**2, dims=(0,)), dim=0)
        tol = 0.1 * energies[-1] * self.options.local_tol ** 2
        
        rank = torch.sum(energies > tol)
        rank = torch.clamp(rank, 1, self.options.max_rank)

        B = U[:, :rank]
        A = (s[:rank] * Vh[:rank].T).T

        if isinstance(poly, Piecewise):
            B = reshape_matlab(B, (poly.cardinality, -1))
            B = torch.linalg.solve(poly.mass_R, B)
            B = reshape_matlab(B, (nn, -1))

        return B, A, rank

    def build_basis_svd(
        self, 
        F: torch.Tensor, 
        k: torch.Tensor|int
    ) -> None:
        """TODO: write docstring..."""

        k = int(k)
        k_prev = int(k - self.data.direction.value)
        k_next = int(k + self.data.direction.value)
        
        poly = self.bases.polys[k]
        interp_x_prev = self.data.interp_x[k_prev]
        core_next = self.data.cores[k_next]

        num_b_left, num_nodes, num_b_right = F.shape 
        rank_0_next, num_nodes_next, rank_1_next = core_next.shape

        # F = reshape_matlab(torch.arange(1, 41*20+1, dtype=torch.float32), (1, 41, 20))

        if self.data.direction == Direction.FORWARD:
            F = torch.permute(F, dims=(1, 0, 2))
            F = reshape_matlab(F, (num_b_left*num_nodes, num_b_right))
            rank_prev = num_b_left
        else: 
            F = torch.permute(F, dims=(1, 2, 0))
            F = reshape_matlab(F, (num_nodes*num_b_right, num_b_left))
            rank_prev = num_b_right

        B, A, rank = self.truncate_local(F, k)

        indices, core, interp_atx = self.select_points(B, k)

        couple = reshape_matlab(interp_atx @ A, (rank, -1))
        interp_x = self.get_local_index(poly, interp_x_prev, indices)

        if self.data.direction == Direction.FORWARD:
            
            core = reshape_matlab(core, (num_nodes, rank_prev, rank))
            core = torch.permute(core, (1, 0, 2))

            couple = torch.permute(couple[:, :rank_0_next], (0, 1))
            couple = reshape_matlab(couple, (-1, rank_0_next))
            
            core_next = reshape_matlab(core_next, (rank_0_next, -1))
            core_next = couple @ core_next
            core_next = reshape_matlab(core_next, (rank, num_nodes_next, rank_1_next))

        else:
            raise NotImplementedError()

        self.data.cores[k] = core
        self.data.interp_x[k] = interp_x 
        self.data.cores[k_next] = core_next

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

    def eval_reference(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the TTFunRef for either the first or last k 
        variables, depending on the current direction the cores are 
        being evaluated in.

        TODO: finish
        """

        fx = self.eval_block(x)
        return fx
    
    def eval_block(
        self, 
        xs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the functional tensor train approximation to the 
        target function for either the first or last k variables, for a 
        set of points in the reference domain.
        
        Parameters
        ----------
        xs:
            A set of points in the reference domain.
        
        Returns
        -------
        fxs:
            The value of the FTT approximation to the target function
            at each point in xs.

        TODO: figure out exactly what's going on in here (I think 
        ultimately it just takes the product of all the cores and basis 
        functions).
            
        """

        num_x, dim_x = xs.shape

        fxs = torch.ones((num_x, 1))

        if self.data.direction == Direction.FORWARD:

            for k in range(min(dim_x, self.bases.dim)):

                rank_p, num_nodes, rank_k = self.data.cores[k].shape

                A_k = self.data.cores[k].permute(1, 0, 2)
                A_k = reshape_matlab(A_k, (num_nodes, -1))

                G_k = self.bases.polys[k].eval_radon(A_k, xs[:, k])
                G_k = reshape_matlab(G_k, (num_x, rank_p, rank_k))
                G_k = G_k.permute(1, 0, 2)
                G_k = reshape_matlab(G_k, (rank_p * num_x, rank_k))

                # TODO: figure out what the below code is doing
                ii = torch.arange(num_x).repeat(rank_p)
                jj = (torch.arange(rank_p * num_x)
                           .reshape(num_x, rank_p).T
                           .flatten())

                indices = torch.vstack((ii[None, :], jj[None, :]))
                size = (num_x, rank_p * num_x)

                B = torch.sparse_coo_tensor(indices, fxs.T.flatten(), size)
                fxs = B @ G_k

        else:
            raise NotImplementedError()

        return fxs.squeeze()

    def grad_reference(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
    
    def int_reference(self):
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

    def _select_points_piecewise(
        self,
        H: torch.Tensor,
        poly: Piecewise
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.options.int_method == "qdeim":
            raise NotImplementedError()

        elif self.options.int_method == "deim":
            raise NotImplementedError()

        elif self.options.int_method == "maxvol":
            indices, core = maxvol(H)
            interp_atx = H[indices]
        
        if (cond := torch.linalg.cond(interp_atx)) > 1e+5:
            msg = f"Poor condition number in interpolation: {cond}."
            warnings.warn(msg)

        return indices, core, interp_atx
    
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
            raise NotImplementedError()  # TODO: learn what this is

        elif self.options.int_method == "maxvol":
            indices, _ = maxvol(nodes)
            interp_atx = nodes[indices]
            core = H @ torch.linalg.inv(interp_atx)
        
        if (cond := torch.linalg.cond(interp_atx)) > 1e+5:
            msg = f"Poor condition number in interpolation ({cond})."
            warnings.warn(msg)

        return indices, core, interp_atx

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

        info(" | ".join(info_headers))
        return

    def _print_info(
        self,
        als_iter: int,
        indices: torch.Tensor
    ) -> None:

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

        info(" | ".join(diagnostics))
        return

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

    def _compute_cross_iter_fixed_rank(
        self, 
        func: Callable,
        indices: torch.Tensor
    ) -> None:

        for k in indices:
                    
            x_left = self.data.interp_x[int(k-1)]
            x_right = self.data.interp_x[int(k+1)]
            
            F = self.build_block_local(func, x_left, x_right, k) 
            self.build_basis_svd(F, k)

        return
    
    def _compute_cross_iter_random(
        self,
        func: Callable,
        indices: torch.Tensor
    ) -> None:
        
        for k in indices:
                
            enrich = self.input_data.get_samples(self.options.kick_rank)
            x_left = self.data.interp_x[int(k-1)]
            x_right = self.data.interp_x[int(k+1)]
            
            # TODO: figure out why enrich seems to use a different set 
            # of samples each time despite the original blocks using 
            # the same sets

            if self.data.direction == Direction.FORWARD:
                enrich = enrich[:, k+1:]
            else:
                enrich = enrich[:, :k]

            F = self.build_block_local(func, x_left, x_right, k)
            F_enrich = self.build_block_local(func, x_left, enrich, k)
            # TODO: check whether the concatenation here is correct.
            F_full = torch.concatenate((F, F_enrich), dim=2)

            self.build_basis_svd(F_full, k)

        return None
    
    def _compute_final_block(self, func: Callable) -> None:

        if self.data.direction == Direction.FORWARD:
            k = self.bases.dim-1 
        else:
            k = 0

        x_left = self.data.interp_x[int(k-1)]
        x_right = self.data.interp_x[int(k+1)]
        self.data.cores[k] = self.build_block_local(func, x_left, x_right, k)

        return

    def initialise_amen(self):
        """TODO: figure out how AMEN works and tidy this up."""

        num_dims = self.bases.dim

        # Initialise the residual coordinates for AMEN
        if self.data.res_x == {}:  # TODO: check what res_x is doing
            # Nested interpolation points for res
            # TODO: the below conditional could probably be tidied up (only the indexing differs in the branches)
            if self.data.direction > 0: # direction has already been fipped
                for k in range(num_dims-1, -1, -1):
                    tmp = self.input_data.get_samples(self.options.kick_rank)
                    self.data.res_x[k] = tmp[:, k:] # TODO: check whether the indexing here is correct
            else:
                for k in range(num_dims-1, -1, -1):
                    tmp = self.input_data.get_samples(self.options.kick_rank)
                    self.data.res_x[k] = tmp[:, :k]

        # TODO: figure out what AMEN is 
        if self.data.res_w == {}:
            if self.data.direction < 0:
                for k in range(num_dims-1):
                    res_shape = (self.options.kick_rank, self.data.cores[k].shape[2])
                    self.data.res_w[k] = torch.ones(res_shape)
            else:
                for k in range(1, num_dims):
                    res_shape = (self.data.cores[k].shape[0], self.options.kick_rank)
                    self.data.res_w[k] = torch.ones(res_shape)

        if self.data.res_w == {}:
            raise NotImplementedError()
        
        return

    def cross(
        self, 
        func: Callable[[torch.Tensor], torch.Tensor]
    ):
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
            self.initialise_cores()
        else:
            # Prepare for the next iteration
            # TODO: figure out why this is here--I think arg (in the 
            # constructor) can be an existing TTFunc to refine.
            self.data.reverse_direction()
        
        if self.use_amen:
            self.initialise_amen()

        while True:

            if self.data.direction == Direction.FORWARD:
                indices = torch.arange(self.bases.dim-1)
            else:
                indices = torch.arange(self.bases.dim-1, 0, -1)
            
            if self.options.tt_method == "fixed_rank":
                self._compute_cross_iter_fixed_rank(func, indices)
            
            elif self.options.tt_method == "random":
                self._compute_cross_iter_random(func, indices)

            elif self.options.tt_method == "amen":
                raise NotImplementedError("AMEN has not been implemented yet.")
            
            else: 
                raise Exception("Unknown TT method.")

            als_iter += 1

            if self.is_finished(als_iter, indices): 
                self._compute_final_block(func)

            self.compute_relative_error()

            if self.is_finished(als_iter, indices):
                self._print_info(als_iter, indices)
                info(f"ALS complete.")
                info(f"Final TT ranks: {[int(r) for r in self.rank]}.")
                return

            else:
                self._print_info(als_iter, indices)
                self.data.reverse_direction()