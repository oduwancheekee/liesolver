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
from .model import LieSolver, BrickFamily
from .plotting import plot_2d_domain, plot_ic_bc, plot_fit_history

class Trainer:
    """Data loading, model initialization, training, and plotting.
    
    Args:
        config: Configuration dict with 'data', 'fit', 'bricks', 'seed' keys.
        out_dir: Output directory for results.
    """
    def __init__(self, config: dict, out_dir=None) -> None: 
        self.config = config
        self.out_dir = out_dir
        print(f"Experiment folder: {self.out_dir}")
        save_config(self.config, self.out_dir)
        self.data_cfg: dict = self.config.get('data', {})
        self.fit_cfg: dict = self.config.get('fit', {})
        self.bricks_cfg: dict = self.config.get('bricks', {})
        self.seed = self.config.get('seed', 0)
        np.random.seed(self.seed)
        self.experiment_name = self.config.get('experiment_name', 'ICBC_type')
        self.state = FitState()
    
    def init_model(self):
        """Initialize DataLoader, BrickFamilies, and LieSolver from config."""
        pde_name = self.config.get('pde', 'pde_type')
        pde_module = importlib.import_module(f"liesolver.pdes.{pde_name}")
        self.trafos = getattr(pde_module, 'trafos')
        self.coords_sym = getattr(pde_module, 'coords_sym')
        
        self.data = DataLoader(pde_name=pde_name, **self.data_cfg)
        families = [BrickFamily.init_from_str(brick_str, self.trafos, self.coords_sym) for brick_str in self.bricks_cfg]

        # Get constraints from data loader
        constraints = self.data.train_constraints
        
        self.model = LieSolver(
            families=families,
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
        max_bricks = self.fit_cfg.get('max_bricks', 20)
        mse_tol = float(self.fit_cfg.get('mse_tol', 1e-3))
        nfev_global = self.fit_cfg.get('nfev_global', 2)
        nfev_batch = self.fit_cfg.get('nfev_batch', 10)
        global_every = self.fit_cfg.get('global_every', 10)
        batch_size = self.fit_cfg.get('batch_size', 5)
        pool_size = self.fit_cfg.get('pool_size', 100)
        
        interrupted = False
        print(f"Action           MSEtrain   MSEtest   MSEdomain Trafos•start_fun         Parameters") 
        try:
            for i in range(max_bricks):
                # Add brick with the highest score
                # Score is cosine similarity with residual 
                score = self.model.add_best_brick(pool_size=pool_size)
                self.state.log(self.model, self.data, step_type='add_best_brick')
                a = self.model.amplitudes[-1]
                sign = '+' if a > 0 else '-'
                if np.abs(a) < 0.01:
                    amp_str = f'a:{sign}{np.format_float_scientific(abs(a), precision=0, exp_digits=1, trim='-')}'
                else:
                    amp_str = f'a:{sign}{np.abs(a):.2f}'
                brick = self.model.bricks[-1]
                print(f"Add {i+1:<2} {amp_str} | {self.state.train_mse_hist[-1]:.2e} | {self.state.test_mse_hist[-1]:.1e} | {self.state.domain_mse_hist[-1]:.1e} | {brick.family} | {brick}")

                # Refine batch - only batch_size of last added bricks             
                K = len(self.model.bricks) 
                if ((i + 1) % batch_size) == 0:
                    active_idx = list(range(max(0, K - batch_size), K))
                    self.model.refine(max_nfev=nfev_batch, active_idx=active_idx)
                    self.state.log(self.model, self.data, step_type='refine_batch')
                    print(f"Refine batch   | {self.model.mse:.2e} | {self.state.test_mse_hist[-1]:.1e} | {self.state.domain_mse_hist[-1]:.1e}")
                
                # Refine all parameters
                if (
                    (((i + 1) % global_every) == 0)
                    or ((i + 1) == max_bricks)
                    # or (self.mse < mse_tol)
                ):
                    self.model.refine(max_nfev=nfev_global)
                    self.state.log(self.model, self.data, step_type='refine_all')
                    print(f"Refine all {K:>2}  | {self.model.mse:.2e} | {self.state.test_mse_hist[-1]:.1e} | {self.state.domain_mse_hist[-1]:.1e}")
                if (self.model.mse <= mse_tol):
                    break
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[Interrupted] Stopping training early. Saving results with {len(self.model.bricks)} bricks...")
        
        self.model.save(self.out_dir)
        self.state.save(self.out_dir / f"{self.experiment_name}-fit_state.npz")
        plot_fit_history(self.state, 
                         save_to = self.out_dir / f"{self.experiment_name}-fit_history.png",
                         compact=False)
        plot_ic_bc(self.model, self.data, 
                   filepath=self.out_dir / f"{self.experiment_name}-ic_bc_plot.png",
                   compact=True)
        plot_ic_bc(self.model, self.data, decompose=True, 
                   filepath=self.out_dir / f"{self.experiment_name}-ic_bc_plot-decompose.png",
                   compact=True)
        plot_2d_domain(self.model, self.data, 
                       filepath=self.out_dir /  f"{self.experiment_name}-domain_plot.png",
                       compact=True)
        
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
    """Tracks history of bricks, parameters, and metrics during training."""
    nbricks_hist: List[float] = field(default_factory=list)
    nparams_hist: List[float] = field(default_factory=list)
    
    train_mse_hist: List[float] = field(default_factory=list)
    test_mse_hist: List[float] = field(default_factory=list)
    domain_mse_hist: List[float] = field(default_factory=list)
    train_l2re_hist: List[float] = field(default_factory=list)
    test_l2re_hist: List[float] = field(default_factory=list)
    domain_l2re_hist: List[float] = field(default_factory=list)
    
    # Detailed history for analysis
    amplitudes_hist: List[np.ndarray] = field(default_factory=list)
    brick_params_hist: List[List[np.ndarray]] = field(default_factory=list)
    step_type_hist: List[str] = field(default_factory=list)
    
    # Custom metrics tracking
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)  # metric_name -> [values]
    _metric_trackers: List[Any] = field(default_factory=list, repr=False)  # List of Metric instances
    
    def set_metrics(self, metrics: List[Any]) -> None:
        """Set custom metrics to track during training.
        
        Args:
            metrics: List of Metric instances to track.
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
        """Log current training state and compute metrics.
        
        Args:
            model: Current LieSolver model.
            data: DataLoader with train/test/domain data.
            step_type: Description of training step ('add_best_brick', 'refine_batch', etc.).
        """
        nparams = sum(brick.params.size for brick in model.bricks)
        self.nparams_hist.append(nparams)
        self.nbricks_hist.append(len(model.bricks))
        self.train_mse_hist.append(model.mse)

        # Compute test MSE respecting derivative orders in constraints
        if hasattr(data, 'test_constraints') and len(data.test_constraints) > 0:
            test_residuals = []
            for c in data.test_constraints:
                F_c = model.eval_bricks(model.bricks, c.x, c.deriv_order)
                pred_c = F_c @ model.amplitudes
                test_residuals.append(np.sqrt(c.weight) * (pred_c - c.y))
            test_r = np.concatenate(test_residuals)
            test_mse = float(np.mean(test_r ** 2))
        else:
            test_mse = float(np.mean((model(data.test_x) - data.test_y)**2))
        test_l2re = np.sqrt(test_mse/np.mean(data.test_y**2)) if np.mean(data.test_y**2) > 0 else 0.0
        domain_mse = float(np.mean((model(data.domain_x) - data.domain_y)**2))
        domain_l2re = np.sqrt(domain_mse/np.mean(data.test_y**2)) if np.mean(data.test_y**2) > 0 else 0.0
        
        self.test_mse_hist.append(test_mse)
        self.test_l2re_hist.append(test_l2re)
        self.domain_mse_hist.append(domain_mse)
        self.domain_l2re_hist.append(domain_l2re)
        
        # Track detailed history
        self.amplitudes_hist.append(model.amplitudes.copy())
        brick_params_snapshot = [brick.params.copy() for brick in model.bricks]
        self.brick_params_hist.append(brick_params_snapshot)
        self.step_type_hist.append(step_type)
        
        # Compute custom metrics
        for metric in self._metric_trackers:
            value = metric.compute(model, data)
            self.metrics_history[metric.name].append(value)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save FitState to NPZ file.
        
        Args:
            filepath: Path to save the state.
        """
        filepath = Path(filepath)
        save_dict = {
            'nbricks_hist': np.array(self.nbricks_hist),
            'nparams_hist': np.array(self.nparams_hist),
            'train_mse_hist': np.array(self.train_mse_hist),
            'test_mse_hist': np.array(self.test_mse_hist),
            'domain_mse_hist': np.array(self.domain_mse_hist),
            'train_l2re_hist': np.array(self.train_l2re_hist),
            'test_l2re_hist': np.array(self.test_l2re_hist),
            'domain_l2re_hist': np.array(self.domain_l2re_hist),
            'amplitudes_hist': np.array(self.amplitudes_hist, dtype=object),
            'brick_params_hist': np.array(self.brick_params_hist, dtype=object),
            'step_type_hist': np.array(self.step_type_hist),
        }
        
        # Save custom metrics
        for metric_name, metric_values in self.metrics_history.items():
            save_dict[f'custom_metric_{metric_name}'] = np.array(metric_values)
        np.savez_compressed(filepath, **save_dict)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FitState':
        """Load FitState from NPZ file.
        
        Args:
            filepath: Path to load the state from.
        
        Returns:
            FitState: Restored state object.
        """
        filepath = Path(filepath)
        data = np.load(filepath, allow_pickle=True)
        
        state = cls()
        state.nbricks_hist = data['nbricks_hist'].tolist()
        state.nparams_hist = data['nparams_hist'].tolist()
        state.train_mse_hist = data['train_mse_hist'].tolist()
        state.test_mse_hist = data['test_mse_hist'].tolist()
        state.domain_mse_hist = data['domain_mse_hist'].tolist()
        state.train_l2re_hist = data['train_l2re_hist'].tolist()
        state.test_l2re_hist = data['test_l2re_hist'].tolist()
        state.domain_l2re_hist = data['domain_l2re_hist'].tolist()
        state.amplitudes_hist = [arr for arr in data['amplitudes_hist']]
        state.brick_params_hist = [list(params) for params in data['brick_params_hist']]
        state.step_type_hist = data['step_type_hist'].tolist()
        
        # Load custom metrics
        for key in data.keys():
            if key.startswith('custom_metric_'):
                metric_name = key.replace('custom_metric_', '')
                state.metrics_history[metric_name] = data[key].tolist()
        
        return state
        

