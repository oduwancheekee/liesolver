from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import hashlib
import importlib
import json

import numpy as np
import sympy as sp
from scipy.stats import qmc

from .constraints import Constraint, ConstraintSet


class BoxDomain:
    """Axis-aligned box domain.
    Args:
        coords_names (Sequence[str]): Coordinate names, e.g., ["x","y","t"].
        bounds (np.ndarray): (dim, 2) bounds per coordinate [low, high].
    """
    def __init__(self, coords_names: Sequence[str], bounds: np.ndarray):
        self.coords_names = list(coords_names)
        self.dim = len(self.coords_names)
        self.bounds = np.asarray(bounds, dtype=float)
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]
        self.width = self.high - self.low
        self.t_idx = self.coords_names.index("t") if "t" in self.coords_names else None
        self.spatial_idxs = [i for i in range(self.dim) if i != self.t_idx]

    def random_domain_points(self, n: int, sampler: str = "halton", seed: Optional[int] = None) -> np.ndarray:
        """Random points in the whole domain.
        Args:
            n (int): Number of points.
            sampler (str): 'random'|'sobol'|'lhs'|'latin'|'latin_hypercube'|'halton'.
            seed (int | None): RNG/QMC seed.
        Returns:
            np.ndarray: (n, dim) points scaled from unit box.
        """
        if sampler == "random":
            unit = np.random.default_rng(seed).random((n, self.dim), dtype=float)
        elif sampler == "sobol":
            unit = qmc.Sobol(d=self.dim, scramble=True, seed=seed).random(n)
        elif sampler in ("lhs", "latin", "latin_hypercube"):
            unit = qmc.LatinHypercube(d=self.dim, seed=seed).random(n)
        elif sampler == "halton":
            unit = qmc.Halton(d=self.dim, scramble=True, seed=seed).random(n)
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")
        return self.low + unit * self.width

    def random_face_points(self, n: int, coord_name: str, bound_idx: int, 
                           sampler: str = "halton", seed: Optional[int] = None) -> np.ndarray:
        if coord_name not in self.coords_names:
            raise ValueError(f"Unknown coordinate: {coord_name}. Available: {self.coords_names}")
        coord_idx = self.coords_names.index(coord_name)
        X = self.random_domain_points(n, sampler, seed)
        X[:, coord_idx] = self.bounds[coord_idx, bound_idx]
        return X

    def random_initial_points(self, n: int, sampler: str = "halton", seed: Optional[int] = None) -> np.ndarray:
        if self.t_idx is None:
            return np.empty((0, self.dim), dtype=float)
        return self.random_face_points(n, "t", 0, sampler, seed)

    def random_boundary_points(self, n: int, sampler: str = "halton", seed: Optional[int] = None) -> np.ndarray:
        faces = [(i, side) for i in self.spatial_idxs for side in (0, 1)]
        if len(faces) == 0:
            return np.empty((0, self.dim), dtype=float)
        rng = np.random.default_rng(seed)
        face_idx = rng.integers(len(faces), size=n)
        X = self.random_domain_points(n, sampler, seed)
        axes = np.array([faces[k][0] for k in face_idx], dtype=int)
        sides = np.array([faces[k][1] for k in face_idx], dtype=int)
        X[np.arange(n), axes] = self.bounds[axes, sides]
        if self.t_idx is not None and self.width[self.t_idx] > 0: # maps t from [t_low, t_high) to (t_low, t_high] 
            u = (X[:, self.t_idx] - self.low[self.t_idx]) / self.width[self.t_idx]
            X[:, self.t_idx] = self.high[self.t_idx] - u * self.width[self.t_idx]
        return X

    def uniform_domain_points(self, n: int) -> np.ndarray:
        """Uniform grid points in the whole domain (includes endpoints).
        Args:
            n (int): Number of points.
        Returns:
            np.ndarray: (m, dim) points from a dim-D grid; may differ from n.
        """
        k = int(np.ceil(n ** (1 / self.dim)))
        grids = [np.linspace(self.low[i], self.high[i], k) for i in range(self.dim)]
        mg = np.meshgrid(*grids, indexing="ij")
        X = np.stack([g.ravel() for g in mg], axis=1)
        if X.shape[0] != n:
            print(f"Warning: requested {n} domain points, returning {X.shape[0]}")
        return X
    
    def uniform_initial_points(self, n: int) -> np.ndarray:
        """Uniform grid points on initial-time face t = low.
        Args:
            n (int): Number of points.
        Returns:
            np.ndarray: (m, dim) points from a (dim-1)-D grid; may differ from n or empty if no t.
        """
        if self.t_idx is None:
            return np.empty((0, self.dim), dtype=float)
        k = int(np.ceil(n ** (1 / len(self.spatial_idxs))))
        grids = [np.linspace(self.low[i], self.high[i], k) for i in self.spatial_idxs]
        mg = np.meshgrid(*grids, indexing="ij")
        S = np.stack([g.ravel() for g in mg], axis=1)
        X = np.empty((S.shape[0], self.dim), dtype=float)
        for col, idx in enumerate(self.spatial_idxs):
            X[:, idx] = S[:, col]
        X[:, self.t_idx] = self.low[self.t_idx]
        if X.shape[0] != n:
            print(f"Warning: requested {n} initial points, returning {X.shape[0]}")
        return X
    
    def uniform_boundary_points(self, n: int) -> np.ndarray:
        """Uniform grid points on spatial boundary for t in (t_low, t_high].
        Args:
            n (int): Number of points.
        Returns:
            np.ndarray: (m, dim) stacked face grids; may differ from n.
        """
        faces = [(i, side) for i in self.spatial_idxs for side in (0, 1)]
        if len(faces) == 0:
            return np.empty((0, self.dim), dtype=float)
        face_dim = self.dim - 1
        n_per_face = int(np.ceil(n / len(faces)))
        k = int(np.ceil(n_per_face ** (1 / face_dim)))
        chunks = []
        for i, side in faces:
            varying_axes = [j for j in range(self.dim) if j != i]
            grids = []
            for j in varying_axes:
                if j == self.t_idx:
                    grids.append(np.linspace(self.low[j], self.high[j], k + 1)[1:])
                else:
                    grids.append(np.linspace(self.low[j], self.high[j], k))
            mg = np.meshgrid(*grids, indexing="ij")
            S = np.stack([g.ravel() for g in mg], axis=1)
            X = np.empty((S.shape[0], self.dim), dtype=float)
            for col, j in enumerate(varying_axes):
                X[:, j] = S[:, col]
            X[:, i] = self.bounds[i, side]
            chunks.append(X)
        X = np.vstack(chunks)
        if X.shape[0] != n:
            print(f"Warning: requested {n} boundary points, returning {X.shape[0]}")
        return X

    def uniform_face_points(self, n: int, coord_name: str, bound_idx: int) -> np.ndarray:
        """Uniform grid points on a face (excludes endpoints to avoid overlap with corners).
        
        Args:
            n (int): Target number of points.
            coord_name (str): Coordinate name fixed on face (e.g., 't' or 'x').
            bound_idx (int): 0 for low bound, 1 for high bound.
        
        Returns:
            np.ndarray: (m, dim) points; may differ from n due to grid rounding.
        """
        if coord_name not in self.coords_names:
            raise ValueError(f"Unknown coordinate: {coord_name}. Available: {self.coords_names}")
        coord_idx = self.coords_names.index(coord_name)
        
        # Dimensions that vary on this face
        varying_axes = [j for j in range(self.dim) if j != coord_idx]
        if len(varying_axes) == 0:
            # 1D domain, face is a single point
            X = np.zeros((1, self.dim))
            X[0, coord_idx] = self.bounds[coord_idx, bound_idx]
            return X
        
        face_dim = len(varying_axes)
        k = int(np.ceil(n ** (1 / face_dim)))
        
        # Create grid for varying dimensions (exclude endpoints to avoid corner overlap)
        grids = [np.linspace(self.low[j], self.high[j], k + 2)[1:-1] for j in varying_axes]
        mg = np.meshgrid(*grids, indexing="ij")
        S = np.stack([g.ravel() for g in mg], axis=1)
        
        X = np.empty((S.shape[0], self.dim), dtype=float)
        for col, j in enumerate(varying_axes):
            X[:, j] = S[:, col]
        X[:, coord_idx] = self.bounds[coord_idx, bound_idx]
        
        if X.shape[0] != n:
            print(f"Warning: requested {n} face points, returning {X.shape[0]}")
        return X


class DataLoader:
    """Analytic PDE data generator with constraint-based IC/BC specification."""

    def __init__(
        self,
        pde_name: str,
        constraints: Sequence[Mapping[str, Any]],
        range_dim: Union[Sequence[Sequence[float]], Mapping[str, Sequence[float]]],
        res: Union[int, Sequence[int]] = 100,
        num_domain: int = 10000,
        test_ratio: float = 1.0,
        phys: Optional[Mapping[str, float]] = None,
        data_dir: Union[str, Path] = "data",
        refresh: Union[bool, int] = False,
        sampler_train: str = "halton",
        sampler_test: str = "uniform",
        pde_mse_tol: float = 1e-10,
        icbc_mse_tol: float = 1e-7,
    ) -> None:
        # Resolve PDE module
        self.pde_name = pde_name
        self._module = importlib.import_module(f"liesolver.pdes.{pde_name}")
        self.coords_sym = getattr(self._module, "coords_sym")
        self._solve_icbc = getattr(self._module, "solve_icbc", None)
        self._get_pde_residual = getattr(self._module, "get_pde_residual", None)
        self._get_icbc_error = getattr(self._module, "get_icbc_error", None)
        
        self._constraints_config = list(constraints)
        self.res = res
        self.num_domain = int(num_domain)
        self.test_ratio = float(test_ratio)
        self.phys: Dict[str, float] = dict(phys) if phys else {}
        self.sampler_train = sampler_train
        self.sampler_test = sampler_test
        self.pde_mse_tol = float(pde_mse_tol)
        self.icbc_mse_tol = float(icbc_mse_tol)

        # Geometry
        self.bounds = self._validate_bounds(range_dim)
        self.geom_dict = self._build_geom_dict()
        self.geometry = BoxDomain([str(s) for s in self.coords_sym], self.bounds)
        
        # Build icbc dict from constraints
        self.icbc = self._icbc_from_constraints()
        
        # Caching
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.data_dir / f"{self.pde_name}-{self._dataset_hash()}.npz"
        
        if Path(self.cache_path).exists() and not bool(refresh):
            self.load(self.cache_path)
            self._build_constraints()
            self._build_test_constraints()
            print(f"Loaded: {self.cache_path} | MSE_pde={self.pde_mse:.2e} | MSE_icbc={self.icbc_mse:.2e}")
        else:
            print(f"Generating reference data with {self.res} modes ...")
            self.u_expr = self._solve_icbc(self.icbc, geom=self.geom_dict, phys=self.phys, res=self.res)
            self.u_func = sp.lambdify(tuple(self.coords_sym), self.u_expr, modules="numpy")
            self._sample_all()
            self.icbc_mse = float(self._get_icbc_error(self.u_expr, self.icbc, geom=self.geom_dict, res=self.res)["MSE"])
            self.pde_mse = float(self._get_pde_residual(self.u_expr, geom=self.geom_dict, phys=self.phys, res=self.res)["MSE"])
            self.save(self.cache_path)
            print(f"Reference data generated | MSE_pde={self.pde_mse:.2e} | MSE_icbc={self.icbc_mse:.2e}")
        
        if self.pde_mse > self.pde_mse_tol:
            print(f"WARNING PDE MSE>{self.pde_mse_tol:.2e}")
        if self.icbc_mse > self.icbc_mse_tol:
            print(f"WARNING ICBC MSE>{self.icbc_mse_tol:.2e}")

    def _validate_bounds(self, range_dim) -> np.ndarray:
        coord_names = [str(s) for s in self.coords_sym]
        if isinstance(range_dim, Mapping):
            arr = np.zeros((len(coord_names), 2), dtype=float)
            for name, bounds in range_dim.items():
                if name not in coord_names:
                    raise ValueError(f"Unknown coordinate '{name}'")
                arr[coord_names.index(name)] = bounds
        else:
            arr = np.asarray(range_dim, dtype=float)
        return arr

    def _build_geom_dict(self) -> Dict[str, float]:
        names = [str(s) for s in self.coords_sym]
        geom = {}
        for i, name in enumerate(names):
            geom[f"{name}_min"] = float(self.bounds[i, 0])
            geom[f"{name}_max"] = float(self.bounds[i, 1])
        return geom

    def _parse_expr(self, expr_raw) -> sp.Expr:
        """Parse constraint expression to sympy."""
        coord_names = [str(s) for s in self.coords_sym]
        locals_map = {n: sp.Symbol(n, real=True) for n in coord_names}
        if isinstance(expr_raw, sp.Expr):
            return expr_raw
        if isinstance(expr_raw, str):
            return sp.parse_expr(expr_raw, transformations="all", local_dict=locals_map)
        return sp.sympify(expr_raw)

    def _icbc_from_constraints(self) -> Dict[str, sp.Expr]:
        """Build icbc dict from constraints config.
        
        Maps constraints to icbc keys based on loc and deriv_order:
        - loc: [t, 0], deriv=(0,0) -> u0 (IC value)
        - loc: [t, 0], deriv=(0,1) -> ut0 (IC velocity)
        - loc: [x, 0], deriv=(0,0) -> bL (BC at x_min)
        - loc: [x, 1], deriv=(0,0) -> bR (BC at x_max)
        """
        t_idx = self.geometry.t_idx
        icbc: Dict[str, sp.Expr] = {}
        
        for cfg in self._constraints_config:
            expr = self._parse_expr(cfg.get('expr', 0))
            loc = cfg.get('loc')
            if loc is None:
                continue
            coord_name, bound_idx = loc[0], int(loc[1])
            deriv_order = cfg.get('deriv_order', [0] * len(self.coords_sym))
            
            if coord_name == 't' and bound_idx == 0:
                # Initial condition
                if t_idx is not None and deriv_order[t_idx] == 1:
                    icbc['ut0'] = expr
                elif all(d == 0 for d in deriv_order):
                    icbc['u0'] = expr
            elif coord_name == 'x':
                # Boundary condition
                if all(d == 0 for d in deriv_order):
                    if bound_idx == 0:
                        icbc['bL'] = expr
                    else:
                        icbc['bR'] = expr
        return icbc

    def _dataset_hash(self) -> str:
        payload = {
            "pde": self.pde_name,
            "coords": [str(s) for s in self.coords_sym],
            "bounds": self.bounds.tolist(),
            "icbc": {k: sp.srepr(v) for k, v in self.icbc.items()},
            "phys": {k: float(v) for k, v in sorted(self.phys.items())},
            "res": list(self.res) if isinstance(self.res, (list, tuple, np.ndarray)) else int(self.res),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def _eval_u(self, X: np.ndarray) -> np.ndarray:
        vals = self.u_func(*[X[:, i] for i in range(X.shape[1])])
        return np.asarray(vals, dtype=float).reshape(-1)
    
    def _eval_expr_at_points(self, expr: sp.Expr, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if expr.is_number or len(expr.free_symbols) == 0:
            return np.full(n, float(expr), dtype=float)
        func = sp.lambdify(self.coords_sym, expr, modules="numpy")
        vals = func(*[X[:, i] for i in range(X.shape[1])])
        return np.full(n, float(vals)) if np.isscalar(vals) else np.asarray(vals, dtype=float).reshape(-1)

    def _build_constraints(self) -> None:
        """Build ConstraintSet from constraints config."""
        geom = self.geometry
        dim = len(self.coords_sym)
        self.train_constraints = ConstraintSet()
        self._train_ic_n = 0
        self._train_bc_n = 0
        
        for i, cfg in enumerate(self._constraints_config):
            expr = self._parse_expr(cfg.get('expr', 0))
            loc = cfg.get('loc')
            if loc is None:
                raise ValueError(f"Constraint {i}: 'loc' is required")
            coord_name, bound_idx = loc[0], int(loc[1])
            deriv_order = tuple(cfg.get('deriv_order', [0] * dim))
            weight = float(cfg.get('weight', 1.0))
            num_samples = int(cfg.get('num_samples', 1000))
            name = cfg.get('name', f"constraint_{i}")
            
            X = geom.random_face_points(num_samples, coord_name, bound_idx, self.sampler_train)
            y = self._eval_expr_at_points(expr, X)
            
            self.train_constraints.add(Constraint(x=X, y=y, deriv_order=deriv_order, weight=weight, name=name))
            
            if coord_name == 't' and bound_idx == 0:
                self._train_ic_n += num_samples
            else:
                self._train_bc_n += num_samples
        
        # Legacy arrays
        if self.train_constraints.n_constraints > 0:
            self.train_x = np.vstack([c.x for c in self.train_constraints])
            self.train_y = np.concatenate([c.y for c in self.train_constraints])
        else:
            self.train_x = np.empty((0, dim))
            self.train_y = np.empty(0)

    def _build_test_constraints(self) -> None:
        """Build test_constraints mirroring train_constraints.
        
        Uses sampler_test to control sampling method:
        - 'uniform': uniform grid points on face (no random seed needed)
        - 'halton'/'sobol'/'random': quasi-random with offset seed for independence
        """
        geom = self.geometry
        dim = len(self.coords_sym)
        
        self.test_constraints = ConstraintSet()
        for i, cfg in enumerate(self._constraints_config):
            expr = self._parse_expr(cfg.get('expr', 0))
            loc = cfg.get('loc')
            coord_name, bound_idx = loc[0], int(loc[1])
            deriv_order = tuple(cfg.get('deriv_order', [0] * dim))
            weight = float(cfg.get('weight', 1.0))
            num_samples = int(cfg.get('num_samples', 1000))
            num_test = int(num_samples * self.test_ratio)
            name = cfg.get('name', f"constraint_{i}")
            
            if self.sampler_test == "uniform":
                # Uniform grid points on face
                X_test = geom.uniform_face_points(num_test, coord_name, bound_idx)
            else:
                # Quasi-random with offset seed for independence from train set
                X_test = geom.random_face_points(num_test, coord_name, bound_idx, 
                                                 sampler=self.sampler_test, seed=42 + i)
            
            # Evaluate target expression (for derivative constraints, this is the derivative target)
            y_test = self._eval_expr_at_points(expr, X_test)
            
            self.test_constraints.add(Constraint(
                x=X_test, y=y_test, deriv_order=deriv_order, weight=weight, name=f"test_{name}"
            ))

    def _sample_all(self) -> None:
        geom = self.geometry
        dim = len(self.coords_sym)
        self._build_constraints()
        self._build_test_constraints()
        
        # Legacy test arrays (stacked from test_constraints for backward compatibility)
        if self.test_constraints.n_constraints > 0:
            self.test_x = np.vstack([c.x for c in self.test_constraints])
            self.test_y = np.concatenate([c.y for c in self.test_constraints])
        else:
            self.test_x = np.empty((0, dim))
            self.test_y = np.empty(0)
        
        self.domain_x = geom.uniform_domain_points(self.num_domain)
        self.domain_y = self._eval_u(self.domain_x)

    def save(self, path: Union[str, Path]) -> None:
        np.savez_compressed(
            str(path),
            train_x=self.train_x, train_y=self.train_y,
            test_x=self.test_x, test_y=self.test_y,
            domain_x=self.domain_x, domain_y=self.domain_y,
            icbc_mse=float(self.icbc_mse), pde_mse=float(self.pde_mse),
            u_expr_str=sp.sstr(self.u_expr), u_expr_srepr=sp.srepr(self.u_expr),
            pde_name=self.pde_name,
            coords_names=np.array([str(s) for s in self.coords_sym], dtype=object),
            bounds=self.bounds, res=np.asarray(self.res if isinstance(self.res, (list, tuple)) else [self.res]),
            phys_json=json.dumps(self.phys), icbc_json=json.dumps({k: sp.sstr(v) for k, v in self.icbc.items()}),
        )

    def load(self, path: Union[str, Path]) -> None:
        data = np.load(path, allow_pickle=True)
        self.train_x, self.train_y = data["train_x"], data["train_y"]
        self.test_x, self.test_y = data["test_x"], data["test_y"]
        self.domain_x, self.domain_y = data["domain_x"], data["domain_y"]
        self.icbc_mse, self.pde_mse = float(data["icbc_mse"]), float(data["pde_mse"])
        
        coords_names = [str(c) for c in data["coords_names"]]
        self.coords_sym = [sp.Symbol(n, real=True) for n in coords_names]
        self.u_expr = sp.parse_expr(str(data["u_expr_str"]), transformations="all", 
                                     local_dict={n: sp.Symbol(n, real=True) for n in coords_names})
        self.u_func = sp.lambdify(tuple(self.coords_sym), self.u_expr, modules="numpy")

    def get_batch(self, split: str = "train", batch_size: Optional[int] = None,
                  seed: Optional[int] = None, shuffle: bool = True, stratify: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        X, y = {"train": (self.train_x, self.train_y), "test": (self.test_x, self.test_y),
                "domain": (self.domain_x, self.domain_y)}[split]
        n = len(X)
        if batch_size is None or batch_size >= n:
            idx = np.arange(n)
            if shuffle:
                np.random.default_rng(seed).shuffle(idx)
            return X[idx], y[idx]
        rng = np.random.default_rng(seed)
        if split == "train" and stratify and (self._train_ic_n + self._train_bc_n) > 0:
            p_ic = self._train_ic_n / (self._train_ic_n + self._train_bc_n)
            m_ic = int(round(batch_size * p_ic))
            ic_idx = rng.integers(0, self._train_ic_n, size=m_ic) if self._train_ic_n else np.array([], dtype=int)
            bc_idx = rng.integers(self._train_ic_n, n, size=batch_size - m_ic) if self._train_bc_n else np.array([], dtype=int)
            idx = np.concatenate([ic_idx, bc_idx])
        else:
            idx = rng.integers(0, n, size=batch_size)
        if shuffle:
            rng.shuffle(idx)
        return X[idx], y[idx]
