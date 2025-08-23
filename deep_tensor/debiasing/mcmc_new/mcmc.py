import math
import time
from typing import Dict, List

import torch
from torch import multiprocessing as mp
from torch import Tensor

from .kernel import Kernel
from ..mcmc import estimate_iact


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
            diagnostics["iacts"][i] = estimate_iact(xs[i])
        
        return


class MCMC():

    def __init__(
        self, 
        kernel: Kernel, 
        num_steps: int,
        x0s: Tensor | List[None] | None = None,
        num_chains: int = 1,
        num_warmup: int = 0
    ):
        """kernel: kernel
        num_steps: number of steps to take after the warmup phase.
        num_chains: number of parallel chains to run.
        num_warmup: number of warmup steps to take

        """

        if x0s is None:
            x0s = [None] * num_chains

        self.kernel = kernel 
        self.num_steps = int(num_steps)
        self.x0s = x0s 
        self.num_chains = num_chains 
        self.num_warmup = num_warmup

        manager = mp.Manager()
        self.diagnostics = manager.dict()
        self.diagnostics["time_per_it"] = manager.dict()
        self.diagnostics["acceptance_rate"] = manager.dict()
        self.diagnostics["iacts"] = manager.dict()
        
        xs_shape = (self.num_chains, self.num_steps, self.kernel.dim)
        potentials_shape = (self.num_chains, self.num_steps)

        self.xs = torch.empty(xs_shape).share_memory_()
        self.potentials = torch.empty(potentials_shape).share_memory_()
        
        return
    
    def run(self):

        # https://docs.pytorch.org/docs/stable/notes/multiprocessing.html#cpu-in-multiprocessing
        n_threads = max(math.floor(mp.cpu_count() / self.num_chains), 1)
        torch.set_num_threads(n_threads)

        print("Running MCMC...")

        processes = []
        for i in range(self.num_chains):
            args = (
                i, 
                self.kernel, 
                self.num_steps, 
                self.x0s[i], 
                self.num_warmup, 
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
        print(self.diagnostics["acceptance_rate"])
        print(self.diagnostics["iacts"])

        import matplotlib.pyplot as plt 
        plt.plot(self.potentials.T)
        plt.show()

        return 

