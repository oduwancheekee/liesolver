from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import hashlib
import importlib
import json

import numpy as np
import sympy as sp
from scipy.stats import qmc


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

    def random_initial_points(self, n: int, sampler: str = "halton", seed: Optional[int] = None) -> np.ndarray:
        """Random points on initial-time face t = t_low.
        Args:
            n (int): Number of points.
            sampler (str): 'random'|'sobol'|'lhs'|'latin'|'latin_hypercube'|'halton'.
            seed (int | None): RNG/QMC seed.
        Returns:
            np.ndarray: (n, dim) points; t fixed at low or empty if no t.
        """
        if self.t_idx is None:
            return np.empty((0, self.dim), dtype=float)
        X = self.random_domain_points(n, sampler, seed)
        X[:, self.t_idx] = self.low[self.t_idx]
        return X

    def random_boundary_points(self, n: int, sampler: str = "halton", seed: Optional[int] = None) -> np.ndarray:
        """Random points on spatial boundary for t in (t_low, t_high].
        Args:
            n (int): Number of points.
            sampler (str): 'random'|'sobol'|'lhs'|'latin'|'latin_hypercube'|'halton'.
            seed (int | None): Seed for engines and face selection.
        Returns:
            np.ndarray: (n, dim) points uniformly per spatial face.
        """
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


class DataLoader:
    """
    Analytic PDE data generator and cache.

    Resolves a PDE module (liesolver.pdes.<pde_name>), builds an analytic solution from IC/BC via solve_icbc,
    samples IC/BC/domain points, computes IC/BC and PDE residual MSEs,
    and caches the dataset to disk. Supports loading cached datasets.

    Args:
        pde_name: Name of the PDE module under liesolver.pdes/ (e.g., "heat_1d", "wave_1d").
        icbc_name: Named IC/BC entry to load from the PDE module's icbcs dict. Ignored if icbc_expr is provided.
        icbc_expr: IC/BC specification overriding icbc_name. Accepts:
            - str or sympy.Expr: interpreted as u0(x,...) for initial condition;
            - dict: keys like "u0", "ut0" mapped to str or sympy.Expr for multi-IC PDEs (e.g., wave).
        range_dim: Bounds for each coordinate strictly matching coords_sym order from the PDE module.
            Example for heat_1d (coords_sym: [x, t]): [[x_min, x_max], [t_min, t_max]].
        res: Series resolution for solve_icbc and residual/error sampling; int or Sequence[int].
        num_ic: Number of initial-condition samples for train/test (time-dependent PDEs).
        num_bc: Number of boundary-condition samples for train/test.
        num_domain: Number of interior/domain samples.
        phys: Optional physical parameters dict passed to PDE solver (e.g., {"alpha": 1.0}).
        data_dir: Directory to save/load datasets.
        refresh: If False/0 and cached file exists, load it; if True/1, recompute and overwrite cache.
        sampler_train: Sampler name for training boundary/initial points (default "halton").
        pde_mse_tol: Threshold for PDE MSE warning (default 1e-10).
        icbc_mse_tol: Threshold for IC/BC MSE warning (default 1e-7).

    Attributes:
        coords_sym: List of SymPy symbols defining coordinate order.
        train_x, train_y: Training inputs/targets.
        test_x, test_y: Testing inputs/targets.
        domain_x, domain_y: Interior inputs/targets.
        u_expr: SymPy expression of the analytic solution.
        u_func: Numpy-callable function u(*coords) from lambdified u_expr.
        icbc_mse: IC/BC mean squared error (float).
        pde_mse: PDE residual mean squared error (float).
        cache_path: Path to the cached dataset file.
    """

    def __init__(
        self,
        pde_name: str,
        icbc_name: Optional[str] = None,
        icbc_expr: Optional[Union[str, sp.Expr, Mapping[str, Union[str, sp.Expr]]]] = None,
        range_dim: Optional[Sequence[Sequence[float]]] = None,
        res: Union[int, Sequence[int]] = 100,
        num_ic: int = 1000,
        num_bc: int = 1000,
        num_domain: int = 10000,
        phys: Optional[Mapping[str, float]] = None,
        data_dir: Union[str, Path] = "data",
        refresh: Union[bool, int] = False,
        sampler_train: str = "halton",
        pde_mse_tol: float = 1e-10,
        icbc_mse_tol: float = 1e-7,
    ) -> None:
        # Resolve PDE module API
        self.pde_name = pde_name
        self._module = importlib.import_module(f"liesolver.pdes.{pde_name}")
        self.coords_sym = getattr(self._module, "coords_sym")
        self._solve_icbc = getattr(self._module, "solve_icbc")
        self._get_pde_residual = getattr(self._module, "get_pde_residual")
        self._get_icbc_error = getattr(self._module, "get_icbc_error")
        self._icbcs = getattr(self._module, "icbcs", {})

        # User parameters
        self.icbc_name = icbc_name
        self.icbc_expr = icbc_expr
        self.res = res
        self.num_ic = int(num_ic)
        self.num_bc = int(num_bc)
        self.num_domain = int(num_domain)
        self.phys: Dict[str, float] = dict(phys) if phys is not None else {}
        self.sampler_train = sampler_train
        self.pde_mse_tol = float(pde_mse_tol)
        self.icbc_mse_tol = float(icbc_mse_tol)

        # Geometry
        self.bounds = self._validate_bounds(range_dim)
        self.geom_dict = self._build_geom_dict()
        self.icbc = self._normalize_icbc(self.icbc_expr, self.icbc_name)
        self.geometry = BoxDomain([str(s) for s in self.coords_sym], self.bounds)
        
        # Caching
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.data_dir / f"{self.pde_name}-{self._dataset_hash()}.npz"

        # Load or compute
        if Path(self.cache_path).exists() and not bool(refresh):
            self.load(self.cache_path)
            print(f"Loaded: {self.cache_path} | MSE_pde={self.pde_mse:.2e} | MSE_icbc={self.icbc_mse:.2e}")
            self._warn_poor_mse()
        else:
            print(f"Generating data with {self.res} modes ...")
            # Solve analytic solution and lambdify
            self.u_expr = self._solve_icbc(self.icbc, geom=self.geom_dict, phys=self.phys, res=self.res)
            self.u_func = sp.lambdify(tuple(self.coords_sym), self.u_expr, modules="numpy")

            # Sample data and metrics
            self._sample_all()
            self.icbc_mse = float(self._get_icbc_error(self.u_expr, self.icbc, geom=self.geom_dict, res=self.res)["MSE"])
            self.pde_mse = float(
                self._get_pde_residual(self.u_expr, geom=self.geom_dict, phys=self.phys, res=self.res)["MSE"]
            )

            # Save
            self.save(self.cache_path)
            print(f"Data generated | MSE_pde={self.pde_mse:.2e} | MSE_icbc={self.icbc_mse:.2e} | saved: {self.cache_path}")
            self._warn_poor_mse()

    # --------------------------- helpers ---------------------------

    def _warn_poor_mse(self) -> None:
        if self.pde_mse > self.pde_mse_tol:
            print(f"WARNING  PDE MSE>{self.pde_mse_tol:.2e}")
        if self.icbc_mse > self.icbc_mse_tol:
            print(f"WARNING ICBC MSE>{self.icbc_mse_tol:.2e}")

    def _validate_bounds(self, range_dim: Sequence[Sequence[float]]) -> np.ndarray:
        if len(range_dim) != len(self.coords_sym):
            raise ValueError(f"range_dim length {len(range_dim)} must match coords_sym length {len(self.coords_sym)}.")
        arr = np.asarray(range_dim, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("range_dim must be a sequence of [min, max] pairs.")
        if np.any(arr[:, 1] <= arr[:, 0]):
            raise ValueError("Each [min, max] must satisfy max > min.")
        return arr

    def _build_geom_dict(self) -> Dict[str, float]:
        """Build a geometry dict with unified keys: x_min/x_max, (y_min/y_max), t_min/t_max based on coords_sym order."""
        names = [str(s) for s in self.coords_sym]
        geom: Dict[str, float] = {}
        for i, name in enumerate(names):
            lo, hi = float(self.bounds[i, 0]), float(self.bounds[i, 1])
            if name == "x":
                geom["x_min"], geom["x_max"] = lo, hi
            elif name == "y":
                geom["y_min"], geom["y_max"] = lo, hi
            elif name == "t":
                geom["t_min"], geom["t_max"] = lo, hi
            else:
                geom[f"{name}_min"], geom[f"{name}_max"] = lo, hi
        return geom

    def _normalize_icbc(
        self,
        icbc_expr: Optional[Union[str, sp.Expr, Mapping[str, Union[str, sp.Expr]]]],
        icbc_name: Optional[str],
    ) -> Dict[str, sp.Expr]:
        """
        Normalize IC/BC specification into a dict of {key: sympy.Expr} (e.g., {'u0': Expr, 'ut0': Expr}).
        icbc_expr overrides icbc_name. Unknown keys are ignored.
        """
        names = [str(s) for s in self.coords_sym]
        locals_map = {n: sp.Symbol(n, real=True) for n in names}

        def _parse(v) -> sp.Expr:
            if isinstance(v, sp.Expr):
                return v
            if isinstance(v, str):
                return sp.parse_expr(v, transformations="all", local_dict=locals_map)
            return sp.parse_expr(str(v), transformations="all", local_dict=locals_map)

        if icbc_expr is not None:
            if isinstance(icbc_expr, Mapping):
                out: Dict[str, sp.Expr] = {}
                for k, v in icbc_expr.items():
                    try:
                        out[str(k)] = _parse(v)
                    except Exception:
                        # ignore unknown/unparseable keys silently
                        pass
                return out
            else:
                return {"u0": _parse(icbc_expr)}

        if icbc_name is not None:
            if icbc_name not in self._icbcs:
                raise KeyError(f"IC/BC name '{icbc_name}' not found in module.icbcs.")
            mapping = self._icbcs[icbc_name]
            out: Dict[str, sp.Expr] = {}
            for k, v in mapping.items():
                out[str(k)] = _parse(v)
            return out

        raise ValueError("Provide either icbc_expr or icbc_name.")

    def _dataset_hash(self) -> str:
        """Create a short hash (first 16 hex chars of SHA-256) from dataset-defining inputs."""
        def expr_to_srepr_map(d: Mapping[str, sp.Expr]) -> Dict[str, str]:
            return {k: sp.srepr(v) for k, v in d.items()}

        payload = {
            "pde": self.pde_name,
            "coords": [str(s) for s in self.coords_sym],
            "bounds": self.bounds.tolist(),
            "icbc": expr_to_srepr_map(self.icbc),
            "phys": {k: float(v) for k, v in sorted(self.phys.items())},
            "res": (list(self.res) if isinstance(self.res, (list, tuple, np.ndarray)) else int(self.res)),
            "num_ic": self.num_ic,
            "num_bc": self.num_bc,
            "num_domain": self.num_domain,
            "sampler_train": self.sampler_train,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def _eval_u(self, X: np.ndarray) -> np.ndarray:
        """Evaluate u_expr at points X (N,D) using coords_sym order. Returns (N,)."""
        vals = self.u_func(*[X[:, i] for i in range(X.shape[1])])
        return np.asarray(vals, dtype=float).reshape(-1)

    def _sample_all(self) -> None:
        """
        Sample train/test/domain points and evaluate u on them.
        Train uses halton sampler; test and domain are sampled uniformly.
        """
        geom = self.geometry

        # Train: IC + BC
        train_ic = geom.random_initial_points(self.num_ic, self.sampler_train)
        train_bc = geom.random_boundary_points(self.num_bc, self.sampler_train)
        self.train_x = np.vstack([train_ic, train_bc])
        self.train_y = self._eval_u(self.train_x)

        # Test: IC + BC (uniform)
        test_ic = geom.uniform_initial_points(self.num_ic)
        test_bc = geom.uniform_boundary_points(self.num_bc)
        self.test_x = np.vstack([test_ic, test_bc])
        self.test_y = self._eval_u(self.test_x)

        # Domain
        self.domain_x = geom.uniform_domain_points(self.num_domain)
        self.domain_y = self._eval_u(self.domain_x)
        
    # --------------------------- public I/O ---------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save dataset to a compressed .npz file.

        Saved fields:
            - Arrays: train_x, train_y, test_x, test_y, domain_x, domain_y
            - Floats: icbc_mse, pde_mse
            - SymPy: u_expr_str (string), u_expr_srepr (string)
            - Metadata: pde_name, coords_names, bounds, res, phys_json, icbc_name, icbc_json,
                        dataset_id, num_ic, num_bc, num_domain
        """
        path = str(path)
        coords_names = np.array([str(s) for s in self.coords_sym], dtype=object)
        bounds = np.asarray(self.bounds, dtype=float)
        res_arr = np.asarray(self.res if isinstance(self.res, (list, tuple, np.ndarray)) else [self.res], dtype=int)
        phys_json = json.dumps({k: float(v) for k, v in self.phys.items()}, sort_keys=True)
        icbc_json = json.dumps({k: sp.sstr(v) for k, v in self.icbc.items()}, sort_keys=True)

        np.savez_compressed(
            path,
            train_x=self.train_x,
            train_y=self.train_y,
            test_x=self.test_x,
            test_y=self.test_y,
            domain_x=self.domain_x,
            domain_y=self.domain_y,
            icbc_mse=float(self.icbc_mse),
            pde_mse=float(self.pde_mse),
            u_expr_str=sp.sstr(self.u_expr),
            u_expr_srepr=sp.srepr(self.u_expr),
            pde_name=self.pde_name,
            coords_names=coords_names,
            bounds=bounds,
            res=res_arr,
            phys_json=phys_json,
            icbc_name=self.icbc_name if self.icbc_name is not None else "",
            icbc_json=icbc_json,
            dataset_id=self._dataset_hash(),
            num_ic=int(self.num_ic),
            num_bc=int(self.num_bc),
            num_domain=int(self.num_domain),
        )

    def load(self, path: Union[str, Path]) -> None:
        """
        Load dataset from a compressed .npz file. Rebuilds u_expr and u_func.
        """
        data = np.load(path, allow_pickle=True)

        # Arrays
        self.train_x = data["train_x"]
        self.train_y = data["train_y"]
        self.test_x = data["test_x"]
        self.test_y = data["test_y"]
        self.domain_x = data["domain_x"]
        self.domain_y = data["domain_y"]

        # Floats
        self.icbc_mse = float(data["icbc_mse"])
        self.pde_mse = float(data["pde_mse"])

        # Metadata
        self.pde_name = str(data["pde_name"])
        coords_names = [str(c) for c in data["coords_names"]]
        self.coords_sym = [sp.Symbol(n, real=True) for n in coords_names]
        self.bounds = np.asarray(data["bounds"], dtype=float)
        self.res = [int(v) for v in np.asarray(data["res"], dtype=int)]
        if len(self.res) == 1:
            self.res = int(self.res[0])
        self.phys = json.loads(str(data["phys_json"]))
        self.icbc_name = str(data["icbc_name"]) or None
        icbc_map = json.loads(str(data["icbc_json"]))
        self.icbc = {
            k: sp.parse_expr(v, transformations="all", local_dict={n: sp.Symbol(n, real=True) for n in coords_names})
            for k, v in icbc_map.items()
        }
        # Counts
        if "num_ic" in data.files:
            self.num_ic = int(data["num_ic"])
        if "num_bc" in data.files:
            self.num_bc = int(data["num_bc"])
        if "num_domain" in data.files:
            self.num_domain = int(data["num_domain"])

        # Rebuild u_expr and u_func
        expr_str = str(data["u_expr_str"])
        self.u_expr = sp.parse_expr(expr_str, transformations="all", local_dict={n: sp.Symbol(n, real=True) for n in coords_names})
        self.u_func = sp.lambdify(tuple(self.coords_sym), self.u_expr, modules="numpy")

    def resample_train(self) -> None:
        """
        Regenerate training samples (IC/BC) and targets using current geometry and sampler.
        Does not recompute u_expr or MSEs and does not save to disk.
        """
        geom = self.geometry
        train_ic = geom.random_initial_points(self.num_ic, self.sampler_train)
        train_bc = geom.random_boundary_points(self.num_bc, self.sampler_train)
        self.train_x = np.vstack([train_ic, train_bc])
        self.train_y = self._eval_u(self.train_x)

    def get_batch(
        self,
        split: str = "train",
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a batch (X, y) from the selected split without modifying internal arrays.

        Args:
            split: 'train' | 'test' | 'domain'.
            batch_size: Number of samples; if None, returns the entire split.
            seed: RNG seed for reproducible sampling.
            shuffle: Shuffle selected indices before returning.
            stratify: Keep IC/BC ratio for 'train' split; ignored for other splits.

        Returns:
            A tuple (X, y) with shapes (N, D) and (N,).
        """
        if split == "train":
            X, y = self.train_x, self.train_y
        elif split == "test":
            X, y = self.test_x, self.test_y
        elif split == "domain":
            X, y = self.domain_x, self.domain_y
        else:
            raise ValueError("split must be one of {'train','test','domain'}.")

        n = len(X)
        if batch_size is None or batch_size >= n:
            idx = np.arange(n, dtype=int)
            if shuffle:
                rng = np.random.default_rng(seed)
                rng.shuffle(idx)
            return X[idx], y[idx]

        rng = np.random.default_rng(seed)
        if split == "train" and stratify and (self._train_ic_n + self._train_bc_n) > 0:
            p_ic = self._train_ic_n / (self._train_ic_n + self._train_bc_n)
            m_ic = int(round(batch_size * p_ic))
            m_bc = batch_size - m_ic
            ic_idx = rng.integers(0, self._train_ic_n, size=m_ic) if self._train_ic_n > 0 else np.array([], dtype=int)
            bc_idx = rng.integers(self._train_ic_n, self._train_ic_n + self._train_bc_n, size=m_bc) if self._train_bc_n > 0 else np.array([], dtype=int)
            idx = np.concatenate([ic_idx, bc_idx])
        else:
            idx = rng.integers(0, n, size=batch_size)

        if shuffle:
            rng.shuffle(idx)
        return X[idx], y[idx]

    def save_summary(self) -> Dict[str, Any]:
        """
        Compact overview of the dataset.

        Returns:
            Dict with PDE name, dims, coords, bounds, res, counts, and MSEs.
        """
        return {
            "pde": self.pde_name,
            "dims": len(self.coords_sym),
            "coords": [str(s) for s in self.coords_sym],
            "bounds": self.bounds.tolist(),
            "res": self.res if isinstance(self.res, int) else list(self.res),
            "num_ic": self.num_ic,
            "num_bc": self.num_bc,
            "num_domain": self.num_domain,
            "icbc_mse": float(self.icbc_mse),
            "pde_mse": float(self.pde_mse),
            "cache_path": str(self.cache_path),
        }