from __future__ import annotations
import sympy as sp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union
import importlib
from pathlib import Path

from .dataloader import DataLoader
from .utils.io import save_config
from .utils.logging import timing
from .model import LieSolver, Base
from .plotting import plot_2d_domain, plot_ic_bc, plot_fit_history

class Trainer:

    def __init__(self, config: dict, out_dir=None) -> None: 
        self.config = config
        self.out_dir = out_dir
        print(f"Experiment folder: {self.out_dir}")
        save_config(self.config, self.out_dir)
        self.data_cfg: dict = self.config.get('data', {})
        self.fit_cfg: dict = self.config.get('fit', {})
        self.base_cfg: dict = self.config.get('base', {})
        self.seed = self.config.get('seed', 0)
        np.random.seed(self.seed)
        self.experiment_name = self.config.get('experiment_name', 'ICBC_type')
    
    def init_model(self):
        pde_name = self.config.get('pde', 'pde_type')
        pde_module = importlib.import_module(f"liesolver.pdes.{pde_name}")
        self.trafos = getattr(pde_module, 'trafos')
        self.coords_sym = getattr(pde_module, 'coords_sym')
        
        self.data = DataLoader(pde_name=pde_name, **self.data_cfg)
        bases = [Base.init_from_str(base_str, self.trafos, self.coords_sym) for base_str in self.base_cfg]

        self.model = LieSolver(
            X=self.data.train_x,
            y=self.data.train_y,
            bases=bases,
            ridge=1e-1,
            sobol_seed=self.seed,
        )

    @timing
    def fit(self):
        max_terms = self.fit_cfg.get('max_terms', 20)
        mse_tol = float(self.fit_cfg.get('mse_tol', 1e-3))
        nfev_global = self.fit_cfg.get('nfev_global', 2)
        nfev_batch = self.fit_cfg.get('nfev_batch', 10)
        global_every = self.fit_cfg.get('global_every', 10)
        batch_size = self.fit_cfg.get('batch_size', 5)
        pool_size = self.fit_cfg.get('pool_size', 100)

        state = FitState()
        print(f"Action           MSE        Add-score  Trafos•seed_fun    Parameters") 
        for i in range(max_terms):
            # Add term with the highest score
            # Score is cosine similarity with residual 
            score = self.model.add_best_term(pool_size=pool_size)
            state.log(self.model, self.data)
            a = self.model.amplitudes[-1]
            sign = '+' if a > 0 else '-'
            if np.abs(a) < 0.01:
                amp_str = f'a:{sign}{np.format_float_scientific(abs(a), precision=0, exp_digits=1, trim='-')}'
            else:
                amp_str = f'a:{sign}{np.abs(a):.2f}'
            term = self.model.terms[-1]
            print(f"Add {i+1:<2} {amp_str} | {self.model.mse:.2e} | {score:.2e} | {term.base} | {term}")

            # Refine batch - only batch_size of last added terms             
            K = len(self.model.terms) 
            if ((i + 1) % batch_size) == 0:
                active_idx = list(range(max(0, K - batch_size), K))
                self.model.refine(max_nfev=nfev_batch, active_idx=active_idx)
                state.log(self.model, self.data)
                print(f"Refine batch   | {self.model.mse:.2e}")
            
            # Refine all parameters
            if (
                (((i + 1) % global_every) == 0)
                or ((i + 1) == max_terms)
                # or (self.mse < mse_tol)
            ):
                self.model.refine(max_nfev=nfev_global)
                state.log(self.model, self.data)
                print(f"Refine all {K:>2}  | {self.model.mse:.2e}")
            if (self.model.mse <= mse_tol):
                break
        
        self.model.save(self.out_dir)
        plot_fit_history(state, 
                         save_to = self.out_dir / f"{self.experiment_name}-fit_history.png")
        plot_ic_bc(self.model, self.data, 
                   filepath=self.out_dir / f"{self.experiment_name}-ic_bc_plot.png")
        plot_ic_bc(self.model, self.data, decompose=True, 
                   filepath=self.out_dir / f"{self.experiment_name}-ic_bc_plot-decompose.png")
        plot_2d_domain(self.model, self.data, 
                       filepath=self.out_dir /  f"{self.experiment_name}-domain_plot.png")            
        # print(f"PDE residual MSE: {pde_residual(self.model):.2e}")
        print(f"Training complete. Model saved to {self.out_dir}")
        
        return state


@dataclass
class FitState:
    nterms_hist: List[float] = field(default_factory=list)
    nparams_hist: List[float] = field(default_factory=list)
    
    train_mse_hist: List[float] = field(default_factory=list)
    test_mse_hist: List[float] = field(default_factory=list)
    domain_mse_hist: List[float] = field(default_factory=list)
    train_l2re_hist: List[float] = field(default_factory=list)
    test_l2re_hist: List[float] = field(default_factory=list)
    domain_l2re_hist: List[float] = field(default_factory=list)


    def log(
        self,
        model: LieSolver,
        data: DataLoader,
    ) -> None:
        nparams = sum(term.params.size for term in model.terms)
        self.nparams_hist.append(nparams)
        self.nterms_hist.append(len(model.terms))
        self.train_mse_hist.append(model.mse)

        test_mse = float(np.mean((model(data.test_x) - data.test_y)**2))
        test_l2re = np.sqrt(test_mse/np.mean(data.test_y**2))
        domain_mse = float(np.mean((model(data.domain_x) - data.domain_y)**2))
        domain_l2re = np.sqrt(domain_mse/np.mean(data.test_y**2))
        
        self.test_mse_hist.append(test_mse)
        self.test_l2re_hist.append(test_l2re)
        self.domain_mse_hist.append(domain_mse)
        self.domain_l2re_hist.append(domain_l2re)
        

