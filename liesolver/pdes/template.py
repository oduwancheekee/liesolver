"""
Template PDE module for use with dataloader.DataLoader and trainer.Trainer
Implement the entities documented below. See pdes/heat_1d.py and pdes/wave_1d.py for concrete examples
"""
from typing import Dict, Mapping, Sequence, Union
import numpy as np
import sympy as sp

# -------------------------------------------------------------------------
# 1) Coordinate symbols
#    - Define the coordinate symbols used by this PDE.
#    - coords_sym order MUST match how points are sampled and passed to lambdify.
#    - Examples:
#        1D time-dependent (e.g., heat, wave): [x, t]
#        2D time-dependent (e.g., wave 2D): [x, y, t]
#        2D steady (e.g., Helmholtz 2D): [x, y]
# -------------------------------------------------------------------------
x, y, t = sp.symbols("x y t", real=True)
coords_sym = [x, t]  # Adjust as needed, e.g., [x, y, t] or [x, y]


# -------------------------------------------------------------------------
# 2) Optional: Named IC/BC examples (used by DataLoader if icbc_name is provided)
#    - Each entry maps a name -> dict of IC/BC components.
#    - Keys depend on the PDE:
#        - Heat-like: {"u0": <Expr(x, ...)>}
#        - Wave-like: {"u0": <Expr(x, ...), "ut0": <Expr(x, ...)>}
#    - Values must be SymPy expressions in the variables from coords_sym.
# -------------------------------------------------------------------------
icbcs: Dict[str, Dict[str, sp.Expr]] = {
    # Examples (modify/remove):
    "example_u0": {"u0": sp.sin(sp.pi * x)},
    "example_u0_ut0": {"u0": sp.sin(sp.pi * x), "ut0": sp.Integer(0)},
}


# -------------------------------------------------------------------------
# 3) Required API: solve_icbc
#    - Build an analytic solution u_expr(x, ...) that satisfies the PDE with the given IC/BC.
#    - Geometry (geom) and physics (phys) dictionaries must be respected.
#    - Geometry keys (unified naming expected by the DataLoader):
#        Spatial:
#          x_min, x_max           for x
#          y_min, y_max           for y (if present)
#          z_min, z_max           for z (if present)
#        Time (if present):
#          t_min, t_max
#      For time-dependent PDEs, ICs are imposed at t = t_min.
#
#    - icbc is either:
#        • a dict like {"u0": Expr, ...} (multi-component IC/BC), or
#        • a single SymPy Expr interpreted as u0.
#
#    - res controls series resolution / internal sampling; see heat_1d/wave_1d examples.
# -------------------------------------------------------------------------
def solve_icbc(
    icbc: Mapping[str, sp.Expr] | sp.Expr,
    geom: Mapping[str, float] | None = None,
    phys: Mapping[str, float] | None = None,
    res: int | Sequence[int] = 100,
) -> sp.Expr:
    """
    Construct an analytic solution u(x, ...) that satisfies the PDE and given IC/BC.

    Args:
        icbc: IC/BC specification.
              - dict form (recommended): keys like "u0", "ut0" (if applicable), values are SymPy Expr.
              - single Expr: interpreted as "u0" for PDEs with a single initial condition.
        geom: Geometry parameters with unified keys, e.g.:
              - x_min, x_max (and optionally y_min, y_max, ...)
              - t_min, t_max (if time-dependent)
        phys: Physical parameters (e.g., {"alpha": 1.0} or {"c": 1.0}).
        res:  Series/spectral resolution or generic resolution parameter (int or sequence).

    Returns:
        SymPy expression u_expr in variables coords_sym.

    Notes:
        - For time-dependent PDEs, initial conditions are applied at t = t_min.
        - For spatial domains not starting at zero, shift coordinates appropriately
          (e.g., use (x - x_min) and domain length a = x_max - x_min).
        - See pdes/heat_1d.py and pdes/wave_1d.py for full reference implementations.
    """
    raise NotImplementedError("Implement solve_icbc for your PDE (see heat_1d.py or wave_1d.py).")


# -------------------------------------------------------------------------
# 4) Required API: get_pde_residual
#    - Return diagnostic stats for the PDE residual r = L[u] (e.g., u_t - alpha*u_xx).
#    - DataLoader uses the 'MSE' entry for caching metadata.
#    - You may also return additional keys (e.g., 'L2', 'Linf').
#    - Sampling should cover the rectangle defined by geom bounds (spatial × time if present).
# -------------------------------------------------------------------------
def get_pde_residual(
    u_expr: sp.Expr,
    geom: Mapping[str, float] | None = None,
    phys: Mapping[str, float] | None = None,
    res: int | Sequence[int] = 100,
) -> dict:
    """
    Compute PDE residual statistics on a uniform grid.

    Args:
        u_expr: Analytic solution (SymPy expression) in coords_sym.
        geom:   Geometry parameters with unified keys:
                - Spatial: x_min/x_max, y_min/y_max, ...
                - Time:    t_min/t_max (if time-dependent)
        phys:   Physical parameters dict for the PDE operator (e.g., alpha, c).
        res:    If int: same resolution per axis.
                If sequence: interpret as [Nx, (Ny,), Nt] depending on PDE dimension.

    Returns:
        dict with at least:
            - 'MSE': Mean squared residual over the grid (float).
          Optionally:
            - 'L2':  L2 norm (float)
            - 'Linf': Infinity norm (float)

    Notes:
        - See pdes/heat_1d.py and pdes/wave_1d.py for implementation patterns:
            • Build residual r_expr symbolically (e.g., u_t - alpha*u_xx).
            • Lambdify r_expr and evaluate on a grid defined by geom and res.
            • Aggregate statistics and return.
    """
    raise NotImplementedError("Implement get_pde_residual for your PDE (see heat_1d.py or wave_1d.py).")


# -------------------------------------------------------------------------
# 5) Required API: get_icbc_error
#    - Return diagnostic stats for initial/boundary condition satisfaction.
#    - DataLoader uses the 'MSE' entry for caching metadata.
#    - ICs are evaluated at t = t_min (if time-dependent).
#    - BCs should follow your PDE's boundary type (e.g., Dirichlet).
# -------------------------------------------------------------------------
def get_icbc_error(
    u_expr: sp.Expr,
    icbc: Mapping[str, sp.Expr] | sp.Expr,
    geom: Mapping[str, float] | None = None,
    res: int | Sequence[int] | None = 100,
) -> dict:
    """
    Compute IC/BC error statistics on uniform samples.

    Args:
        u_expr: Analytic solution (SymPy expression) in coords_sym.
        icbc:   IC/BC mapping (e.g., {'u0': Expr, 'ut0': Expr}) or a single Expr interpreted as 'u0'.
        geom:   Geometry parameters with unified keys:
                - Spatial: x_min/x_max, y_min/y_max, ...
                - Time:    t_min/t_max (if time-dependent)
        res:    If int: same resolution per axis.
                If sequence: interpret as [Nx, (Ny,), Nt] depending on PDE dimension.

    Returns:
        dict with at least:
            - 'MSE': Mean squared IC/BC error over all evaluated conditions (float).
          Optionally more granular keys, such as:
            - 'IC_MSE', 'BC0_MSE', 'BC1_MSE', etc. (names as appropriate for the PDE).

    Notes:
        - For time-dependent problems, initial conditions are checked at t = t_min.
        - For boundary conditions, evaluate along the spatial boundary as per your PDE.
        - See pdes/heat_1d.py and pdes/wave_1d.py for implementation patterns.
    """
    raise NotImplementedError("Implement get_icbc_error for your PDE (see heat_1d.py or wave_1d.py).")