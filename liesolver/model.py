from __future__ import annotations
import sympy as sp
import numpy as np

from typing import Callable, List, Sequence, Tuple, Union, Optional

from scipy.optimize import least_squares

import re
from pathlib import Path

class Transformation:
    """
    SymPy-based transformation T_eps: Expr -> Expr, with a single scalar parameter eps.
    Stores numeric bounds for eps and provides simple uniform sampling.
    Args:
        kernel        : callable (expr: sp.Expr, eps: sp.Symbol|sp.Expr|float) -> sp.Expr
        param_bounds  : [low, high] numeric bounds for eps (used for candidate sampling and NLLS bounds)
        name          : optional human-readable name (defaults to kernel.__name__)
    """

    def __init__(
        self,
        kernel: Callable[[sp.Expr, Union[sp.Symbol, sp.Expr, float]], sp.Expr],
        param_bounds: Sequence[float],
        idx: int,
        name: Optional[str] = None,
        sample: str = 'uniform',    
    ):
        assert len(param_bounds) == 2, "param_bounds must be [low, high]"
        self.kernel = kernel
        self.param_bounds = (float(param_bounds[0]), float(param_bounds[1]))
        self.idx = idx
        self.name = name or getattr(kernel, "__name__", "transformation")
        self.sample = sample

    def __call__(self, expr: sp.Expr, param: Union[str, float, sp.Expr]) -> sp.Expr:
        if isinstance(param, str):
            eps = sp.Symbol(param, real=True)
        else:
            eps = param
        return self.kernel(expr, eps)

    def __repr__(self):
        return f"<Transformation {self.idx}: {self.kernel.__name__}>"

    def index(self):
        return self.kernel.__name__.replace('_', '')
    
    
class Base:
    """
    base expression = consecutive trafos applied to seed solution function"""
    def __init__(self,
                 seed_solution: sp.Expr,
                 trafos: List[Transformation],
                 coords_sym: List[sp.Symbol]
                 ):
        self.seed_solution = seed_solution
        self.trafos = trafos if isinstance(trafos, list) else [trafos]
        self.coords_sym = coords_sym
        self.params_sym = []
        self.params_bounds = []
        self.param_sample = []
        self.apply_transform()

    def __repr__(self):
        trafos_repr = ''
        for trafo in self.trafos:
            trafos_repr = f'T{trafo.idx}•' + trafos_repr
        return f"{trafos_repr[:-1]}( {self.seed_solution} )"

    def apply_transform(self):
        """
        generates:
        params_sym List
        expression sp.Expr
        f_np function
        param_bounds List[List]
        """
        self.expression = self.seed_solution
        for i in range(len(self.trafos)):
            trafo = self.trafos[i]
            param_name = f'θ{trafo.idx}_{i}'
            param_sym = sp.Symbol(param_name, real=True)
            self.params_sym.append(param_sym)
            self.params_bounds.append(trafo.param_bounds)
            self.param_sample.append(trafo.sample)
            self.expression = trafo(self.expression, param_sym)

        self.f_np = sp.lambdify(self.coords_sym + self.params_sym, self.expression, modules="numpy")

    def eval(self, X: np.ndarray, params: np.ndarray):
        return self.f_np(*X.T, *params)

    @classmethod
    def init_from_str(cls,
                      base_str: str,
                      trafo_catalog: List[Transformation],
                      coords_sym: List[sp.Symbol],
                      ) -> Base:
        """
        parses base str (e.g. T1 T3(0,1,log) exp(x)), 
        e.g. T3(0,1,log) -> will be trafo_catalog[2] with log sampling from 0 to 1
 
        constructs sp.Expr and list[Transformation] 
        inits Base
        """
        matches = list(re.finditer(r"\bT(\d+)(?:\(([^)]*)\))?", base_str))
        trafos_tokens = [(int(m.group(1)), m.group(2)) for m in matches]
        expr_str = base_str[matches[-1].end():].strip() if matches else base_str.strip()

        expr = sp.parse_expr(expr_str, transformations="all", local_dict={s.name: s for s in coords_sym})
        
        catalog_idx = [trafo.idx for trafo in trafo_catalog]
        assert len(catalog_idx) == len(set(catalog_idx)), "idx attributes of default trafos must be unique"
        trafos: List[Transformation] = []
        for idx, override in reversed(trafos_tokens):
            if idx not in catalog_idx:
                raise KeyError(f"Unknown transformation index T{idx}")
            trafo = trafo_catalog[catalog_idx.index(idx)]
            if not override:
                trafos.append(trafo)
            else:
                parts = override.split(',')
                bounds = [float(p) for p in parts[:2]]
                sample = parts[2].strip() if len(parts) == 3 else 'uniform'
                trafo_updated = Transformation(kernel=trafo.kernel,
                                               idx=trafo.idx,
                                               name=trafo.name,
                                               param_bounds=bounds,
                                               sample=sample)
                trafos.append(trafo_updated)

        return cls(seed_solution=expr, trafos=trafos, coords_sym=coords_sym)

    def to_string(self) -> str:
        """
        Serialize this Base into a config string:
        e.g., 'T4(0, 10000, log) T1(0, 3.14, uniform) sin(x) exp(-t)'
        """
        t_parts = []
        for trafo in reversed(self.trafos):  # outermost -> innermost
            low, high = trafo.param_bounds
            t_parts.append(f"T{trafo.idx}({low:.15g}, {high:.15g}, {trafo.sample})")

        expr_str = sp.sstr(self.seed_solution)
        return " ".join(t_parts + [expr_str])

    def sample_params(self, pool_size: int, sampler: str = 'random', sobol_seed=0) -> np.ndarray:
        """
        Samples parameters uniformly within their [low, high] bounds.

        Parameters
        ----------
        sampler : {'random', 'sobol'}
            'random' for pseudorandom (NumPy), 'sobol' for quasirandom Sobol.

        Returns
        -------
        np.ndarray
            Shape (pool_size, len(self.params_bounds))
        """
        n_params = len(self.params_bounds)
        if n_params == 0:
            return np.empty((pool_size, 0))

        if sampler == 'sobol':
            from scipy.stats import qmc
            engine = qmc.Sobol(d=n_params, scramble=True, seed=sobol_seed)
            # Use power-of-two fast path when possible
            if pool_size > 0 and (pool_size & (pool_size - 1)) == 0:
                U = engine.random_base2(m=int(np.log2(pool_size)))
            else:
                U = engine.random(pool_size)
        else:
            U = np.random.uniform(0.0, 1.0, size=(pool_size, n_params))

        sample = []
        for i, (low, high) in enumerate(self.params_bounds):
            if self.param_sample[i] == 'uniform':
                param_pool = low + (high - low) * U[:, i]
            elif self.param_sample[i] == 'log':
                param_pool = low * (high / low) ** U[:, i]
            else:
                raise ValueError(f"Unknown sampling type: {self.param_sample[i]}")
            sample.append(param_pool)

        return np.column_stack(sample)


class BaseTerm:

    def __init__(self, base: Base, params, base_idx: int = None):
        self.base = base
        self.params: np.ndarray = params
        self.base_idx = base_idx

    def __repr__(self):
        params_names = [sym.name.split('_')[0] for sym in self.base.params_sym]
        params_str = ' '.join([f'{params_names[i]}:{self.params[i]:.1e}' for i in range(len(params_names))])
        return f"{params_str}"


class LieSolver:
    
    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            bases: List[Base],
            ridge=0.0,
            sobol_seed=0,
        ):
        self.X = X.astype(np.float64)
        self.y = y.astype(np.float64)
        self.bases = bases
        self.ridge = ridge

        self.terms: List[BaseTerm] = []
        self.amplitudes = np.array([])
        self.mse = np.inf
        self.sobol_seed = sobol_seed

    def __call__(self, X: np.ndarray):
        A = self.design_matrix(self.terms, X)
        return A @ self.amplitudes

    def save(self, folder: Path = None, filename = None) -> None:
        if filename is None:
            filename = f'model_Nbases{len(self.bases)}_Nterms{len(self.terms)}_mse{self.mse:.0e}.npz'
        if folder is None:
            folder = Path.cwd()
        path = Path(folder) / filename
        np.savez_compressed(
            path,
            X = self.X,
            y = self.y,
            bases_str = [base.to_string() for base in self.bases],
            bases_idx = [term.base_idx for term in self.terms],
            theta = np.array([term.params for term in self.terms], dtype=object),
            ridge = self.ridge,
            sobol_seed = self.sobol_seed,
            a = self.amplitudes,
            mse = self.mse,
        )

    @classmethod
    def init_from_npz(cls, file, trafos, coords_sym):
        data = np.load(file, allow_pickle=True)
        model = cls(
            X=data["X"],
            y=data["y"],
            bases=[Base.init_from_str(base_str, trafos, coords_sym) for base_str in data["bases_str"]],
            ridge=float(np.asarray(data["ridge"]).item()),
            sobol_seed=int(np.asarray(data["sobol_seed"]).item()),
        )
        bases_idx = data["bases_idx"]
        theta = data["theta"]
        for i, base_idx in enumerate(bases_idx):
            model.add_defined_term(int(base_idx), np.asarray(theta[i]))
        A = model.design_matrix(model.terms)
        model.amplitudes = model.ls_amplitudes(A)
        r = (A @ model.amplitudes - model.y)
        model.mse = float(np.mean(r ** 2))
        data.close()
        return model

    def add_defined_term(self, base_idx, params):
        term = BaseTerm(self.bases[base_idx],
                        params=params,
                        base_idx=base_idx)
        self.terms.append(term)
    
    def remove_term(self, term_idx: int = -1):
        if not self.terms:
            return None

        if term_idx < 0:
            term_idx += len(self.terms)
        if term_idx < 0 or term_idx >= len(self.terms):
            raise IndexError("term_idx out of range")

        removed = self.terms.pop(term_idx)

        if self.terms:
            A = self.design_matrix(self.terms)
            self.amplitudes = self.ls_amplitudes(A)
            r = A @ self.amplitudes - self.y
            self.mse = float(np.mean(r ** 2))
        else:
            self.amplitudes = np.zeros(0)
            self.mse = float(np.mean(self.y ** 2))
        return removed

    def design_matrix(self, terms: List[BaseTerm], X: np.ndarray = None) -> np.ndarray:
        if X is None:
            X = self.X
        cols = [term.base.eval(X, term.params) for term in terms]
        return np.stack(cols, axis=1)  # (N, K)

    def ls_amplitudes(self, A: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        if y is None:
            y = self.y
        if A.shape[1] == 0:
            return np.zeros(0)
        if self.ridge > 0.0:
            AtA = A.T @ A
            Aty = A.T @ y
            M = AtA.shape[0]
            a = np.linalg.solve(AtA + self.ridge * np.eye(M), Aty)
        else:
            a, *_ = np.linalg.lstsq(A, y, rcond=None)
        return a

    def add_best_term(self, pool_size: int = 1000):
        if len(self.terms) == 0:
            r = self.y.copy()
        else:
            A = self.design_matrix(self.terms)
            a = self.ls_amplitudes(A)
            r = self.y - A @ a
        norm_r = np.linalg.norm(r) + 1e-12
        best_score = -np.inf
        best_base = None

        for base_idx, base in enumerate(self.bases):
            params_pool = base.sample_params(pool_size, self.sobol_seed)  # (pool_size, len(base.params))
            for p in params_pool:
                phi = base.eval(self.X, p)
                norm_phi = np.linalg.norm(phi) + 1e-12
                score = abs(np.dot(r, phi) / norm_phi / norm_r)
                if score > best_score:
                    best_score = score
                    best_base = BaseTerm(base, p.copy(), base_idx)
        const_score = abs(np.dot(r, np.ones_like(r)) / np.linalg.norm(np.ones_like(r)) / norm_r)
        if best_score < const_score:
            print(f'Add log: best score {best_score:.2e} is lower than constant score {const_score:.2e}')
        self.terms.append(best_base)
        
        A = self.design_matrix(self.terms)
        self.amplitudes = self.ls_amplitudes(A)
        r = (A @ self.amplitudes - self.y)
        mse = float(np.mean(r ** 2))
        if mse > self.mse:
            print('Add log: MSE worsened')
        self.mse = mse
        return best_score
        

    def _pack_theta(self, terms: List[BaseTerm]) -> np.ndarray:
        if not terms:
            return np.zeros(0)
        return np.concatenate([term.params for term in terms], axis=0)

    def _unpack_theta(self, theta: np.ndarray, terms: List[BaseTerm], active_idx) -> None:
        """
        Distribute a flat parameter vector theta across bases with variable-length params.
        Expects each Base.params to be a 1D array (size = number of params for that base).
        """
        idx = 0
        for i, base in enumerate(terms):
            if i in active_idx or not active_idx:
                n = int(np.size(base.params))
                base.params = np.asarray(theta[idx: idx + n]).copy()
                idx += n
        assert idx == theta.size, f"theta length {theta.size} != total params consumed {idx}"

    def _bounds_vector(self, terms: List[BaseTerm]) -> Tuple[np.ndarray, np.ndarray]:
        lbs, ubs = [], []
        for term in terms:
            for bounds in term.base.params_bounds:
                lb, ub = bounds
                lbs.append(lb)
                ubs.append(ub)
        if not lbs:
            return np.zeros(0), np.zeros(0)
        return np.array(lbs), np.array(ubs)

    def _residual_varpro(self, theta: np.ndarray, active_idx: List[int]) -> np.ndarray:
        terms = [BaseTerm(term.base, term.params.copy()) for term in self.terms]
        self._unpack_theta(theta, terms, active_idx)
        A = self.design_matrix(terms)
        a = self.ls_amplitudes(A)
        return (A @ a - self.y)
    
    def refine(self, max_nfev: int = 10, active_idx: List[int] = []):
        if active_idx:
            active_terms = [self.terms[i] for i in active_idx]
        else:
            active_terms = self.terms

        res = least_squares(
            fun=lambda th: self._residual_varpro(th, active_idx),
            x0=self._pack_theta(active_terms),
            bounds=self._bounds_vector(active_terms),
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            verbose=0,
        )
        self._unpack_theta(res.x, self.terms, active_idx)
        A = self.design_matrix(self.terms)
        self.amplitudes = self.ls_amplitudes(A)
        r = A @ self.amplitudes - self.y
        mse = float(np.mean(r ** 2))
        if mse > self.mse:
            print('Refine log: MSE worsened')
        elif mse == self.mse:
            print('Refine log: MSE stalled')
        self.mse = mse



