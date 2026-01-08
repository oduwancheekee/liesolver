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
    """Data loading, model initialization, training, saving, and plotting."""
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
        self.state = FitState()
    
    def init_model(self):
        pde_name = self.config.get('pde', 'pde_type')
        pde_module = importlib.import_module(f"liesolver.pdes.{pde_name}")
        self.trafos = getattr(pde_module, 'trafos')
        self.coords_sym = getattr(pde_module, 'coords_sym')
        
        self.data = DataLoader(pde_name=pde_name, **self.data_cfg)
        bases = [Base.init_from_str(base_str, self.trafos, self.coords_sym) for base_str in self.base_cfg]

        # Get constraints from data loader
        constraints = self.data.train_constraints
        
        self.model = LieSolver(
            bases=bases,
            constraints=constraints,
            ridge=1e-1,
            sobol_seed=self.seed,
        )
        
        # Log constraint info
        if constraints is not None and len(constraints) > 0:
            print(constraints.summary())
        

    @timing
    def fit(self):
        """Run greedy add-refine training loop, save model, and generate plots.
        Returns:
            FitState: Collected training metrics.
        """
        max_terms = self.fit_cfg.get('max_terms', 20)
        mse_tol = float(self.fit_cfg.get('mse_tol', 1e-3))
        nfev_global = self.fit_cfg.get('nfev_global', 2)
        nfev_batch = self.fit_cfg.get('nfev_batch', 10)
        global_every = self.fit_cfg.get('global_every', 10)
        batch_size = self.fit_cfg.get('batch_size', 5)
        pool_size = self.fit_cfg.get('pool_size', 100)

        print(f"Action           MSE        Add-score  Trafos•seed_fun    Parameters") 
        for i in range(max_terms):
            # Add term with the highest score
            # Score is cosine similarity with residual 
            score = self.model.add_best_term(pool_size=pool_size)
            self.state.log(self.model, self.data, step_type='add_best_term')
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
                self.state.log(self.model, self.data, step_type='refine_batch')
                print(f"Refine batch   | {self.model.mse:.2e}")
            
            # Refine all parameters
            if (
                (((i + 1) % global_every) == 0)
                or ((i + 1) == max_terms)
                # or (self.mse < mse_tol)
            ):
                self.model.refine(max_nfev=nfev_global)
                self.state.log(self.model, self.data, step_type='refine_all')
                print(f"Refine all {K:>2}  | {self.model.mse:.2e}")
            if (self.model.mse <= mse_tol):
                break
        
        self.model.save(self.out_dir)
        self.state.save(self.out_dir / f"{self.experiment_name}-fit_state.npz")
        plot_fit_history(self.state, 
                         save_to = self.out_dir / f"{self.experiment_name}-fit_history.png")
        plot_ic_bc(self.model, self.data, 
                   filepath=self.out_dir / f"{self.experiment_name}-ic_bc_plot.png")
        plot_ic_bc(self.model, self.data, decompose=True, 
                   filepath=self.out_dir / f"{self.experiment_name}-ic_bc_plot-decompose.png")
        plot_2d_domain(self.model, self.data, 
                       filepath=self.out_dir /  f"{self.experiment_name}-domain_plot.png")
        
        # Plot custom metrics
        from .plotting import plot_custom_metric
        for metric_name in self.state.metrics_history:
            plot_custom_metric(self.state, metric_name,
                             filepath=self.out_dir / f"{self.experiment_name}-metric_{metric_name}.png",
                             log_scale=True)
            
        # print(f"PDE residual MSE: {pde_residual(self.model):.2e}")
        print(f"Training complete. Model saved to {self.out_dir}")
        
        return self.state


@dataclass
class FitState:
    """Tracks history of terms, parameters, and metrics during training.
    """
    nterms_hist: List[float] = field(default_factory=list)
    nparams_hist: List[float] = field(default_factory=list)
    
    train_mse_hist: List[float] = field(default_factory=list)
    test_mse_hist: List[float] = field(default_factory=list)
    domain_mse_hist: List[float] = field(default_factory=list)
    train_l2re_hist: List[float] = field(default_factory=list)
    test_l2re_hist: List[float] = field(default_factory=list)
    domain_l2re_hist: List[float] = field(default_factory=list)
    
    # Detailed history for analysis
    amplitudes_hist: List[np.ndarray] = field(default_factory=list)
    term_params_hist: List[List[np.ndarray]] = field(default_factory=list)
    step_type_hist: List[str] = field(default_factory=list)
    
    # Custom metrics tracking
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)  # metric_name -> [values]
    _metric_trackers: List[Any] = field(default_factory=list, repr=False)  # List of Metric instances
    
    def set_metrics(self, metrics: List[Any]) -> None:
        """Set custom metrics to track during training.
        
        Args:
            metrics (List[Metric]): List of Metric instances to track.
        """
        self._metric_trackers = metrics
        for metric in metrics:
            self.metrics_history[metric.name] = []
    
    def log(
        self,
        model: LieSolver,
        data: DataLoader,
        step_type: str = 'unknown',
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
        
        # Track detailed history
        self.amplitudes_hist.append(model.amplitudes.copy())
        term_params_snapshot = [term.params.copy() for term in model.terms]
        self.term_params_hist.append(term_params_snapshot)
        self.step_type_hist.append(step_type)
        
        # Compute custom metrics
        for metric in self._metric_trackers:
            value = metric.compute(model, data)
            self.metrics_history[metric.name].append(value)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save FitState to npz file.
        
        Args:
            filepath (Union[str, Path]): Path to save the state.
        """
        filepath = Path(filepath)
        save_dict = {
            'nterms_hist': np.array(self.nterms_hist),
            'nparams_hist': np.array(self.nparams_hist),
            'train_mse_hist': np.array(self.train_mse_hist),
            'test_mse_hist': np.array(self.test_mse_hist),
            'domain_mse_hist': np.array(self.domain_mse_hist),
            'train_l2re_hist': np.array(self.train_l2re_hist),
            'test_l2re_hist': np.array(self.test_l2re_hist),
            'domain_l2re_hist': np.array(self.domain_l2re_hist),
            'amplitudes_hist': np.array(self.amplitudes_hist, dtype=object),
            'term_params_hist': np.array(self.term_params_hist, dtype=object),
            'step_type_hist': np.array(self.step_type_hist),
        }
        
        # Save custom metrics
        for metric_name, metric_values in self.metrics_history.items():
            save_dict[f'custom_metric_{metric_name}'] = np.array(metric_values)
        np.savez_compressed(filepath, **save_dict)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FitState':
        """Load FitState from npz file.
        
        Args:
            filepath (Union[str, Path]): Path to load the state from.
        
        Returns:
            FitState: Restored state object.
        """
        filepath = Path(filepath)
        data = np.load(filepath, allow_pickle=True)
        
        state = cls()
        state.nterms_hist = data['nterms_hist'].tolist()
        state.nparams_hist = data['nparams_hist'].tolist()
        state.train_mse_hist = data['train_mse_hist'].tolist()
        state.test_mse_hist = data['test_mse_hist'].tolist()
        state.domain_mse_hist = data['domain_mse_hist'].tolist()
        state.train_l2re_hist = data['train_l2re_hist'].tolist()
        state.test_l2re_hist = data['test_l2re_hist'].tolist()
        state.domain_l2re_hist = data['domain_l2re_hist'].tolist()
        state.amplitudes_hist = [arr for arr in data['amplitudes_hist']]
        state.term_params_hist = [list(params) for params in data['term_params_hist']]
        state.step_type_hist = data['step_type_hist'].tolist()
        
        # Load custom metrics
        for key in data.keys():
            if key.startswith('custom_metric_'):
                metric_name = key.replace('custom_metric_', '')
                state.metrics_history[metric_name] = data[key].tolist()
        
        return state
        

