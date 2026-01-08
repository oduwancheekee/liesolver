import numpy as np
import sympy as sp
from typing import Mapping, Sequence
from scipy.fft import dst

from ..model import Transformation

x, t = sp.symbols('x t', real=True)
coords_sym = [x, t]

def shift_x(expr: sp.Expr, eps):
    return expr.subs(x, x - eps)

def shift_t(expr: sp.Expr, eps):
    return expr.subs(t, t + eps)

def scale_u(expr: sp.Expr, eps):
    return eps * expr

def scale_xt(expr: sp.Expr, eps):
    return expr.subs({x: eps * x, t: eps**2 * t})

def diffusion(expr: sp.Expr, eps):
    denom = 1 + 4 * eps * t
    factor = 1 / sp.sqrt(denom)
    exp_part = sp.exp(-eps * x**2 / denom)
    return factor * exp_part * expr.subs({x: x/denom, t: t/denom})

trafos: list[Transformation] = [
    Transformation(kernel=shift_x, idx=1, param_bounds=[0, 1]),
    Transformation(kernel=shift_t, idx=2, param_bounds=[-10, 10]),
    Transformation(kernel=scale_u, idx=3, param_bounds=[0, 10]),
    Transformation(kernel=scale_xt, idx=4, param_bounds=[0, 100]),
    Transformation(kernel=diffusion, idx=6, param_bounds=[1e-1, 1e6], sample='log'),
]


def solve_icbc(icbc: Mapping[str, sp.Expr],
               geom: Mapping[str, float] | None = None,
               phys: Mapping[str, float] | None = None,
               res: int | Sequence[int] = 100) -> sp.Expr:
    """1D heat equation: u_t = alpha * u_xx with general Dirichlet BCs.
    
    IC: u(x, t_min) = u0(x)
    BC: u(x_min, t) = bL(t), u(x_max, t) = bR(t)
    
    Uses time-dependent lift: w(x,t) = bL(t) + (bR(t) - bL(t)) * (x - x_min) / a
    Then v = u - w satisfies homogeneous Dirichlet BCs.
    """
    x_min = geom.get("x_min", 0.0) if geom else 0.0
    x_max = geom.get("x_max", 1.0) if geom else 1.0
    t_min = geom.get("t_min", 0.0) if geom else 0.0
    alpha = phys.get("alpha", 1.0) if phys else 1.0
    M = res if isinstance(res, int) else int(res[0])
    a = x_max - x_min
    
    u0_expr = icbc.get("u0", sp.Integer(0))
    
    # BC expressions (may depend on t)
    bL_expr = icbc.get("bL")
    bR_expr = icbc.get("bR")
    
    # If BCs not specified, derive from u0 at boundaries (constant BCs)
    if bL_expr is None:
        bL_expr = u0_expr.subs(x, x_min) if not u0_expr.is_zero else sp.Integer(0)
    if bR_expr is None:
        bR_expr = u0_expr.subs(x, x_max) if not u0_expr.is_zero else sp.Integer(0)
    
    # Lift function: w(x,t) = bL(t) + (bR(t) - bL(t)) * (x - x_min) / a
    w = bL_expr + (bR_expr - bL_expr) * (x - x_min) / a
    
    # v0 = u0 - w(x, t_min)
    w_at_t0 = w.subs(t, t_min)
    v0_expr = sp.simplify(u0_expr - w_at_t0)
    
    # Source term from lift: f(x,t) = -w_t + alpha * w_xx
    w_t = sp.diff(w, t)
    w_xx = sp.diff(w, x, 2)
    source = -w_t + alpha * w_xx  # This should be 0 for linear-in-x lift with constant BCs
    
    # Project v0 onto sine modes
    xi = x_min + (np.arange(1, M + 1) * a) / (M + 1)
    v0_samp = np.asarray(sp.lambdify(x, v0_expr, "numpy")(xi), dtype=float)
    F = dst(v0_samp, type=1, norm="ortho")
    scale = np.sqrt(2.0 / (M + 1))
    
    # Homogeneous solution: v(x,t) = sum_m A_m * exp(-alpha*k_m^2*(t-t_min)) * sin(k_m*(x-x_min))
    u_expr = w
    for m in range(1, M + 1):
        k = m * np.pi / a
        A_m = float(scale * F[m - 1])
        u_expr += A_m * sp.exp(-alpha * k**2 * (t - t_min)) * sp.sin(m * sp.pi * (x - x_min) / a)
    
    return u_expr


def get_pde_residual(u_expr: sp.Expr,
                     geom: Mapping[str, float] | None = None,
                     phys: Mapping[str, float] | None = None,
                     res: int | Sequence[int] = 100) -> dict:
    """PDE residual: r = u_t - alpha*u_xx."""
    x_min, x_max = (geom or {}).get("x_min", 0.0), (geom or {}).get("x_max", 1.0)
    t_min, t_max = (geom or {}).get("t_min", 0.0), (geom or {}).get("t_max", 1.0)
    alpha = (phys or {}).get("alpha", 1.0)
    Nx, Nt = (res, res) if isinstance(res, int) else (int(res[0]), int(res[1]))

    r_expr = sp.diff(u_expr, t) - alpha * sp.diff(u_expr, x, 2)
    rf = sp.lambdify((x, t), r_expr, "numpy")
    X, T = np.meshgrid(np.linspace(x_min, x_max, Nx), np.linspace(t_min, t_max, Nt), indexing="xy")
    R = rf(X, T)
    return {"MSE": float(np.mean(R**2)), "Linf": float(np.max(np.abs(R)))}


def get_icbc_error(u_expr: sp.Expr,
                   icbc: Mapping[str, sp.Expr],
                   geom: Mapping[str, float] | None = None,
                   res: int | Sequence[int] = 100) -> dict:
    """IC/BC error for heat equation."""
    x_min, x_max = (geom or {}).get("x_min", 0.0), (geom or {}).get("x_max", 1.0)
    t_min, t_max = (geom or {}).get("t_min", 0.0), (geom or {}).get("t_max", 1.0)
    Nx, Nt = (res, res) if isinstance(res, int) else (int(res[0]), int(res[1]))

    u0_expr = icbc.get("u0", sp.Integer(0))
    bL_expr = icbc.get("bL", u0_expr.subs(x, x_min))
    bR_expr = icbc.get("bR", u0_expr.subs(x, x_max))
    
    u_fn = sp.lambdify((x, t), u_expr, "numpy")
    f_fn = sp.lambdify(x, u0_expr, "numpy")
    bL_fn = sp.lambdify(t, bL_expr, "numpy") if not bL_expr.is_number else lambda _: float(bL_expr)
    bR_fn = sp.lambdify(t, bR_expr, "numpy") if not bR_expr.is_number else lambda _: float(bR_expr)

    xi = np.linspace(x_min, x_max, Nx)
    ti = np.linspace(t_min, t_max, Nt)

    ic_err = u_fn(xi, t_min) - f_fn(xi)
    bc0_err = u_fn(x_min, ti) - bL_fn(ti)
    bca_err = u_fn(x_max, ti) - bR_fn(ti)

    return {
        "IC_MSE": float(np.mean(ic_err**2)),
        "BC0_MSE": float(np.mean(bc0_err**2)),
        "BC1_MSE": float(np.mean(bca_err**2)),
        "MSE": float((np.sum(ic_err**2) + np.sum(bc0_err**2) + np.sum(bca_err**2)) / (len(xi) + 2*len(ti))),
    }
