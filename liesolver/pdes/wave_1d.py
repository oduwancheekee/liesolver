import numpy as np
import sympy as sp
from typing import Mapping, Sequence
from scipy.fft import dst

from ..model import Transformation

x,t = sp.symbols('x t', real=True)
coords_sym = [x, t]

def shift_x(expr: sp.Expr, eps):
    return expr.subs(x, x - eps)

def shift_t(expr: sp.Expr, eps):
    return expr.subs(t, t + eps)

def scale_u(expr: sp.Expr, eps):
    return eps * expr

def scale_xt(expr: sp.Expr, eps):
    return expr.subs({x: eps*x, t: eps*t})



trafos: list[Transformation] = [
    Transformation(kernel=shift_x,
                      idx = 1,
                      param_bounds=[0, 1]),
    Transformation(kernel=shift_t,
                      idx = 2,
                      param_bounds=[-10, 10],
                      ),
    Transformation(kernel=scale_xt,
                      idx = 4,
                      param_bounds=[0, 1000],
                      ),
]

icbcs = {
    "gauss": {'u0': sp.exp(-50 * (x - 0.5) ** 2)},
    "gauss_mix": {'u0': 0.7*sp.exp(-100 * (x - 0.4) ** 2) + sp.exp(-500 * (x - 0.7) ** 2)},
    "sine": {'u0': sp.sin(4*sp.pi*x)},
    "sine_mix": {'u0': 0.5*sp.sin(2*sp.pi*x) - 0.2*sp.sin(4*sp.pi*x)+ 0.7*sp.sin(12*sp.pi*x)},
    "step": {'u0': 0.5 * (sp.tanh(500 * (x - 0.4)) - sp.tanh(500 * (x - 0.6)))},
    # "wave_mix": {'u0': 0.7*sp.exp(-100 * (x - 0.4) ** 2) + sp.exp(-500 * (x - 0.7) ** 2)},
}

def solve_icbc(icbc: Mapping[str, sp.Expr] | sp.Expr,
               geom: Mapping[str, float] | None = None,
               phys: Mapping[str, float] | None = None,
               res: int | Sequence[int] = 100
               ) -> sp.Expr:
    """
    1D wave equation: u_tt = c^2 * u_xx on (x_min, x_max)
    BC: Dirichlet homogeneous u(x_min,t)=u(x_max,t)=0
    IC at t = t_min: u(x, t_min) = u0(x), u_t(x, t_min) = ut0(x)

    Method: project ICs u0 (and optional ut0) onto sine modes via orthonormal DST-I on the interior grid,
            assemble an M-mode series:
            u(x,t)=Σ [A_m cos(ω_m (t - t_min)) + (B_m/ω_m) sin(ω_m (t - t_min))] sin(mπ (x - x_min)/a), 
            with ω_m=c mπ/a and (A_m,B_m) from the DST-I scaled.

    Args: 
        icbc: {'u0': Expr, 'ut0': Expr(optional)} or u0 Expr
        geom: {'a':1.0}
        phys: {'c':1.0}
        res: M (default 100)

    Returns: SymPy Expr u(x,t)
    """
    x_min = 0.0 if geom is None else float(geom.get("x_min", 0.0))
    x_max = 1.0 if geom is None else float(geom.get("x_max", 1.0))
    t_min = 0.0 if geom is None else float(geom.get("t_min", 0.0))
    a = x_max - x_min

    c = 1.0 if phys is None else float(phys.get("c", 1.0))
    M = res if isinstance(res, int) else int(res[0])
    if isinstance(icbc, Mapping):
        u0_expr = icbc.get("u0")
        ut0_expr = icbc.get("ut0", None)
    else:
        u0_expr, ut0_expr = icbc, None

    xi = x_min + (np.arange(1, M + 1) * a) / (M + 1)

    if u0_expr is None or u0_expr == 0 or (isinstance(u0_expr, sp.Expr) and u0_expr.is_zero):
        f_samp = np.zeros(M)
    else:
        f_val = sp.lambdify(x, u0_expr, "numpy")(xi)
        f_samp = np.full(M, f_val) if np.isscalar(f_val) else np.asarray(f_val, dtype=float)
    
    if ut0_expr is None or ut0_expr == 0 or (isinstance(ut0_expr, sp.Expr) and ut0_expr.is_zero):
        g_samp = np.zeros_like(f_samp)
    else:
        g_val = sp.lambdify(x, ut0_expr, "numpy")(xi)
        g_samp = np.full_like(f_samp, g_val) if np.isscalar(g_val) else np.asarray(g_val, dtype=float)

    F = dst(f_samp, type=1, norm="ortho")
    G = dst(g_samp, type=1, norm="ortho")
    scale = np.sqrt(2.0 / (M + 1))

    u_expr = sp.Integer(0)
    for m in range(1, M + 1):
        omega = c * m * np.pi / a
        A_m = float(scale * F[m - 1])
        B_m = float(scale * G[m - 1])
        u_expr += (A_m * sp.cos(omega * (t - t_min)) + (B_m / omega) * sp.sin(omega * (t - t_min))) * sp.sin(
            m * np.pi * (x - x_min) / a
        )
    return u_expr


def get_pde_residual(u_expr: sp.Expr,
                geom: Mapping[str, float] | None = None,
                phys: Mapping[str, float] | None = None,
                res: int | Sequence[int] | None = 100
                ) -> dict:
    """
    Wave 1d PDE residual:
        r(x,t) = u_tt - c^2*u_xx on (x_min,x_max) x (t_min,t_max).
    Args:
        geom: {'x_min':0.0,'x_max':1.0,'t_min':0.0,'t_max':1.0}
        phys: {'c':1.0}
        res: N or [Nx, Nt] (sampling for MSE)
    Returns: {'MSE':..., 'L2':..., 'Linf':...}.
    """
    x_min = 0.0 if geom is None else float(geom.get("x_min", 0.0))
    x_max = 1.0 if geom is None else float(geom.get("x_max", 1.0))
    t_min = 0.0 if geom is None else float(geom.get("t_min", 0.0))
    t_max = 1.0 if geom is None else float(geom.get("t_max", 1.0))
    c = 1.0 if phys is None else float(phys.get("c", 1.0))
    Nx, Nt = (res, res) if isinstance(res, int) else (int(res[0]), int(res[1]))

    r_expr = sp.diff(u_expr, t, 2) - (c**2) * sp.diff(u_expr, x, 2)
    rf = sp.lambdify((x, t), r_expr, "numpy")
    xi = np.linspace(x_min, x_max, Nx)
    ti = np.linspace(t_min, t_max, Nt)
    X, T = np.meshgrid(xi, ti, indexing="xy")
    R = rf(X, T)
    dx = (x_max - x_min) / max(Nx - 1, 1)
    dt = (t_max - t_min) / max(Nt - 1, 1)
    return {"MSE": float(np.mean(R**2)), 
            "L2": float(np.sqrt(np.sum(R**2) * dx * dt)), 
            "Linf": float(np.max(np.abs(R)))}

def get_icbc_error(u_expr: sp.Expr,
               icbc: Mapping[str, sp.Expr] | sp.Expr,
               geom: Mapping[str, float] | None = None,
               res: int | Sequence[int] | None = 100
               ) -> dict:
    """
    IC/BC MSE for wave 1d PDE
    IC at t = t_min: u(x,t_min)=u0(x), u_t(x,t_min)=ut0(x)
    BC: u(x_min,t)=u(x_max,t)=0 (Dirichlet homogeneous)
    Args:
        icbc: {'u0':Expr, 'ut0':Expr(optional)} or u0: Expr
        geom: {'x_min':0.0,'x_max':1.0,'t_min':0.0,'t_max':1.0}
        res: N or [Nx, Nt] (sampling)
    Returns: {'IC_u_MSE':..., 'IC_ut_MSE':..., 'BC0_MSE':..., 'BCa_MSE':..., 'MSE':...}
    """
    x_min = 0.0 if geom is None else float(geom.get("x_min", 0.0))
    x_max = 1.0 if geom is None else float(geom.get("x_max", 1.0))
    t_min = 0.0 if geom is None else float(geom.get("t_min", 0.0))
    t_max = 1.0 if geom is None else float(geom.get("t_max", 1.0))
    Nx, Nt = (res, res) if isinstance(res, int) else (int(res[0]), int(res[1]))
    if isinstance(icbc, Mapping):
        u0_expr = icbc.get("u0")
        ut0_expr = icbc.get("ut0", None)
    else:
        u0_expr, ut0_expr = icbc, None

    u_fn = sp.lambdify((x, t), u_expr, "numpy")
    du_dt = sp.lambdify((x, t), sp.diff(u_expr, t), "numpy")
    f_fn = sp.lambdify((x,), u0_expr, "numpy")
    g_fn = (lambda z: np.zeros_like(z)) if ut0_expr is None else sp.lambdify((x,), ut0_expr, "numpy")

    xi = np.linspace(x_min, x_max, Nx)
    ti = np.linspace(t_min, t_max, Nt)

    ic_u_err = u_fn(xi, t_min) - f_fn(xi)
    ic_ut_err = du_dt(xi, t_min) - g_fn(xi)
    bc0_err = u_fn(x_min, ti)  # target 0
    bca_err = u_fn(x_max, ti)  # target 0

    n_all = xi.size + xi.size + ti.size + ti.size
    total = (np.sum(ic_u_err**2) + np.sum(ic_ut_err**2) + np.sum(bc0_err**2) + np.sum(bca_err**2)) / n_all
    return {
        "IC_u_MSE": float(np.mean(ic_u_err**2)),
        "IC_ut_MSE": float(np.mean(ic_ut_err**2)),
        "BC0_MSE": float(np.mean(bc0_err**2)),
        "BCa_MSE": float(np.mean(bca_err**2)),
        "MSE": float(total),
    }