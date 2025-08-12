import math
import time
from typing import Dict, List

import torch
from torch import multiprocessing as mp
from torch import Tensor

from .kernel import Kernel


lock = mp.Lock()


class _Chain():

    def run(
        self,
        i: int,
        kernel: Kernel, 
        n_steps: int,
        x0: Tensor | None, 
        n_warmup: int,
        xs, 
        potentials,
        diagnostics: Dict
    ):

        t0 = time.time()

        # TODO: this should actually be r0
        kernel._initialise(x0)
        
        for _ in range(n_warmup):
            kernel._step()
        
        for j in range(n_steps):
            print(f"{i} {j}")
            xs[i, j, :], potentials[i, j] = kernel._step()

        time_per_it = (time.time() - t0) / (n_warmup + n_steps)

        with lock:
            diagnostics["acceptance_rate"][i] = kernel.acceptance_rate
            diagnostics["time_per_it"][i] = time_per_it
        
        return


class MCMC():

    def __init__(
        self, 
        kernel: Kernel, 
        n_steps: int,
        x0s: Tensor | List[None] | None = None,
        n_chains: int = 1,
        n_warmup: int = 0
    ):
        """kernel: kernel
        n_warmup: number of warmup steps to take
        n_chains: number of parallel chains to run.
        n_steps: number of steps to take after the warmup phase.
        
        """

        if x0s is None:
            x0s = [None] * n_chains

        self.kernel = kernel 
        self.n_steps = int(n_steps)
        self.x0s = x0s 
        self.n_chains = n_chains 
        self.n_warmup = n_warmup

        manager = mp.Manager()
        self.diagnostics = manager.dict()
        self.diagnostics["time_per_it"] = manager.dict()
        self.diagnostics["acceptance_rate"] = manager.dict()
        
        self.xs = torch.empty((self.n_chains, self.n_steps, self.kernel.dim)).share_memory_()
        self.potentials = torch.empty((self.n_chains, self.n_steps)).share_memory_()
        
        return
    
    def run(self):

        # https://docs.pytorch.org/docs/stable/notes/multiprocessing.html#cpu-in-multiprocessing
        n_threads = min(math.floor(mp.cpu_count()/self.n_chains), 1)
        torch.set_num_threads(n_threads)

        processes = []
        for i in range(self.n_chains):
            args = (
                i, 
                self.kernel, 
                self.n_steps, 
                self.x0s[i], 
                self.n_warmup, 
                self.xs, 
                self.potentials, 
                self.diagnostics
            )
            chain = _Chain()
            p = mp.Process(
                target=chain.run, 
                args=args
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

        print(self.potentials)
        print(self.diagnostics)

        return 

