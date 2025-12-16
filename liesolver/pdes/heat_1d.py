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

def galilean(expr: sp.Expr, eps):
    return sp.exp(-eps * x + eps**2 * t) * expr.subs(x, x - 2*eps*t)

def diffusion(expr: sp.Expr, eps):
    denom = 1 + 4 * eps * t
    factor = 1 / sp.sqrt(denom)
    exp_part = sp.exp(-eps * x**2 / denom)
    return factor * exp_part * expr.subs({x: x/denom, t: t/denom})

trafos: list[Transformation] = [
    Transformation(kernel=shift_x,
                   idx=1,
                   param_bounds=[0, 1]
                   ),
    Transformation(kernel=shift_t,
                   idx=2,
                   param_bounds=[-10, 10],
                   ),
    Transformation(kernel=scale_u,
                   idx=3,
                   param_bounds=[0, 10],
                   ),
    Transformation(kernel=scale_xt,
                   idx=4,
                   param_bounds=[0, 100],
                   ),
    Transformation(kernel=galilean,
                   idx=5,
                   param_bounds=[-1, 1],
                   ),
    Transformation(kernel=diffusion,
                   idx=6,
                   param_bounds=[1e-1, 1e6],
                   sample='log'
                   ),
]

icbcs = {
    "poly": {'u0': x**2 + x**3 - x**5 + x**7},
    "gauss": {'u0': sp.exp(-5 * (x - 0.5) ** 2)},
    # "asym_gauss": {'u0': sp.exp(-3 * (x) ** 2)},
    "sine": {'u0': sp.sin(4*sp.pi*x)},
    "sine_mix": {'u0': 0.5*sp.sin(2*sp.pi*x) - 0.2*sp.sin(4*sp.pi*x)+ 0.7*sp.sin(12*sp.pi*x)},
    "step": {'u0': 0.5 * (sp.tanh(500 * (x - 0.4)) - sp.tanh(500 * (x - 0.6)))},
}


def solve_icbc(icbc: Mapping[str, sp.Expr] | sp.Expr,
               geom: Mapping[str, float] | None = None,
               phys: Mapping[str, float] | None = None,
               res: int | Sequence[int] = 100
               ) -> sp.Expr:
    """
    1D heat equation: u_t = alpha * u_xx on (x_min, x_max)
    IC at t = t_min: u(x, t_min) = u0(x)
    BC: Dirichlet constant u(x_min, t) = u0(x_min), u(x_max, t) = u0(x_max)

    Method: lift u0 by w(x) to homogeneous Dirichlet, project v0=u0-w onto sine modes
            via orthonormal DST-I, assemble M-mode series:
            u(x,t) = w(x) + Σ A_m e^{-alpha (mπ/a)^2 (t - t_min)} sin(mπ (x - x_min)/a)
            where a = x_max - x_min.

    Args:
        icbc: {'u0': Expr} or u0 Expr
        geom: {'x_min':0.0,'x_max':1.0,'t_min':0.0}
        phys: {'alpha':1.0}
        res:  M (default 100)

    Returns:
        SymPy Expr u(x,t)
    """
    x_min = 0.0 if geom is None else float(geom.get("x_min", 0.0))
    x_max = 1.0 if geom is None else float(geom.get("x_max", 1.0))
    t_min = 0.0 if geom is None else float(geom.get("t_min", 0.0))
    alpha = 1.0 if phys is None else float(phys.get("alpha", 1.0))
    M = res if isinstance(res, int) else int(res[0])
    a = x_max - x_min
    u0_expr = icbc.get("u0") if isinstance(icbc, Mapping) else icbc

    bL = float(u0_expr.subs(x, x_min))
    bR = float(u0_expr.subs(x, x_max))
    w = bL + (bR - bL) * (x - x_min) / a
    v0_expr = sp.simplify(u0_expr - w)

    xi = x_min + (np.arange(1, M + 1) * a) / (M + 1)
    v0_samp = np.asarray(sp.lambdify(x, v0_expr, "numpy")(xi), dtype=float)
    F = dst(v0_samp, type=1, norm="ortho")
    scale = np.sqrt(2.0 / (M + 1))
    
    u_expr = w
    for m in range(1, M + 1):
        k = m * np.pi / a
        A_m = float(scale * F[m - 1])
        u_expr += A_m * sp.exp(-alpha * k * k * (t - t_min)) * sp.sin(m * sp.pi * (x - x_min) / a)
    return u_expr


def get_pde_residual(u_expr: sp.Expr,
                geom: Mapping[str, float] | None = None,
                phys: Mapping[str, float] | None = None,
                res: int | Sequence[int] = 100
                ) -> dict:
    """
    Heat 1d PDE residual: r(x,t) = u_t - alpha*u_xx on (x_min,x_max) x (t_min,t_max)

    Args:
        geom: {'x_min':0.0,'x_max':1.0,'t_min':0.0,'t_max':1.0}
        phys: {'alpha':1.0}
        res: N or [Nx, Nt] (sampling for MSE)
    Returns:
        {'MSE':..., 'L2':..., 'Linf':...}
    """
    x_min = 0.0 if geom is None else float(geom.get("x_min", 0.0))
    x_max = 1.0 if geom is None else float(geom.get("x_max", 1.0))
    t_min = 0.0 if geom is None else float(geom.get("t_min", 0.0))
    t_max = 1.0 if geom is None else float(geom.get("t_max", 1.0))
    alpha = 1.0 if phys is None else float(phys.get("alpha", 1.0))
    Nx, Nt = (res, res) if isinstance(res, int) else (int(res[0]), int(res[1]))

    r_expr = sp.diff(u_expr, t) - alpha * sp.diff(u_expr, x, 2)
    rf = sp.lambdify((x, t), r_expr, "numpy")

    xi = np.linspace(x_min, x_max, Nx)
    ti = np.linspace(t_min, t_max, Nt)
    X, T = np.meshgrid(xi, ti, indexing="xy")
    R = rf(X, T)

    dx = (x_max - x_min) / max(Nx - 1, 1)
    dt = (t_max - t_min) / max(Nt - 1, 1)
    return {
        "MSE": float(np.mean(R**2)),
        "L2": float(np.sqrt(np.sum(R**2) * dx * dt)),
        "Linf": float(np.max(np.abs(R))),
    }

def get_icbc_error(u_expr: sp.Expr,
               icbc: Mapping[str, sp.Expr] | sp.Expr,
               geom: Mapping[str, float] | None = None,
               res: int | Sequence[int] | None = 100) -> dict:
    """
    IC/BC MSE for heat 1d PDE
    IC at t = t_min: u(x, t_min) = u0(x)
    BC: u(x_min,t)=u0(x_min), u(x_max,t)=u0(x_max)

    Inputs:
        icbc: {'u0':Expr} or u0 Expr
        geom: {'x_min':0.0,'x_max':1.0,'t_min':0.0,'t_max':1.0}
        res:  N or [Nx, Nt] (sampling)
    Returns:
        {'IC_MSE':..., 'BC0_MSE':..., 'BC1_MSE':..., 'MSE':...}
    """
    x_min = 0.0 if geom is None else float(geom.get("x_min", 0.0))
    x_max = 1.0 if geom is None else float(geom.get("x_max", 1.0))
    t_min = 0.0 if geom is None else float(geom.get("t_min", 0.0))
    t_max = 1.0 if geom is None else float(geom.get("t_max", 1.0))
    Nx, Nt = (res, res) if isinstance(res, int) else (int(res[0]), int(res[1]))

    u0_expr = icbc.get("u0") if isinstance(icbc, Mapping) else icbc
    u_fn = sp.lambdify((x, t), u_expr, "numpy")
    f_fn = sp.lambdify((x,), u0_expr, "numpy")
    bL, bR = float(u0_expr.subs(x, x_min)), float(u0_expr.subs(x, x_max))

    xi = np.linspace(x_min, x_max, Nx)
    ti = np.linspace(t_min, t_max, Nt)

    ic_err = u_fn(xi, t_min) - f_fn(xi)
    bc0_err = u_fn(x_min, ti) - bL
    bca_err = u_fn(x_max, ti) - bR

    n_all = xi.size + ti.size + ti.size
    total = (np.sum(ic_err**2) + np.sum(bc0_err**2) + np.sum(bca_err**2)) / n_all
    return {
        "IC_MSE": float(np.mean(ic_err**2)),
        "BC0_MSE": float(np.mean(bc0_err**2)),
        "BC1_MSE": float(np.mean(bca_err**2)),
        "MSE": float(total),
    }