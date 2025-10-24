import numpy as np
import sympy as sp
from typing import Mapping, Sequence
from scipy.fft import dst

from ..model import Transformation

x, y, t = sp.symbols('x y t', real=True)
coords_sym = [x, y, t]

def shift_x(expr: sp.Expr, eps):
    return expr.subs(x, x - eps)

def shift_t(expr: sp.Expr, eps):
    return expr.subs(t, t + eps)

def scale_u(expr: sp.Expr, eps):
    return eps * expr

def hyperbolic_rotation_x(expr: sp.Expr, eps):
    """
    Lorentz boost mixing (x,t) with rapidity 'eps':
        x' = ch x - sh t
        t' = -sh x + ch t
    """
    ch = sp.cosh(eps); sh = sp.sinh(eps)
    return expr.subs({x: ch*x - sh*t, t: -sh*x + ch*t}, simultaneous=True)

def scale_xt(expr: sp.Expr, eps):
    """
    Scale in spacetime: (x,t) -> (s x, s t) with s = e^eps.
    """
    # s = sp.exp(eps)
    return expr.subs({x: eps*x, t: eps*t})

def inversion_x(expr: sp.Expr, eps):
    """
    Special conformal transformation in +x direction.
    """
    D = 1 + 2*eps*x - eps**2 * (t**2 - x**2)
    mapped = expr.subs({x: (x - eps*(t**2 - x**2)) / D, t:  t / D}, simultaneous=True)
    return mapped / sp.sqrt(D)


def inversion_t(expr: sp.Expr, eps):
    """
    Special conformal transformation in +t direction.
    """
    D = 1 + 2*eps*t + eps**2 * (t**2 - x**2)
    mapped = expr.subs({x:  x / D, t:  (t + eps*(t**2 - x**2)) / D}, simultaneous=True)
    return mapped / sp.sqrt(D)



trafos: list[Transformation] = [
    Transformation(kernel=shift_x,
                      idx = 1,
                      param_bounds=[0, 1]),
    Transformation(kernel=shift_t,
                      idx = 2,
                      param_bounds=[-10, 10],
                      ),
    Transformation(kernel=hyperbolic_rotation_x,
                      idx = 3,
                      param_bounds=[0, 10],
                      ),
    Transformation(kernel=scale_xt,
                      idx = 4,
                      param_bounds=[0, 1000],
                      ),
    Transformation(kernel=inversion_x,
                      idx = 5,
                      param_bounds=[-1, 1],
                      ),
    Transformation(kernel=inversion_t,
                      idx = 6,
                      param_bounds=[1e-1, 1e6],
                    #   sample = 'log'
                      ),
]

def solve_icbc(
    icbc: Mapping[str, sp.Expr] | sp.Expr,
    geom: Mapping[str, float] | None = None,
    phys: Mapping[str, float] | None = None,
    res: int | Sequence[int] = 100,
) -> sp.Expr:
    """
    2D wave equation: u_tt = c^2 * (u_xx + u_yy) on (x_min, x_max) x (y_min, y_max)
    BC: Dirichlet homogeneous u(x_min, y, t)=u(x_max, y, t)=u(x, y_min, t)=u(x, y_max, t)=0
    IC at t = t_min: u(x, y, t_min) = u0(x, y), u_t(x, y, t_min) = ut0(x, y)

    Method:
        - Project ICs u0 (and optional ut0) onto 2D sine modes via separable orthonormal DST-I on the interior grid.
        - Assemble series:
          u(x,y,t) = Σ_{m=1..Mx} Σ_{n=1..My} [A_{m,n} cos(ω_{m,n} (t - t_min)) + (B_{m,n}/ω_{m,n}) sin(ω_{m,n} (t - t_min))]
                      * sin(mπ (x - x_min)/a) * sin(nπ (y - y_min)/b),
          where a = x_max - x_min, b = y_max - y_min, ω_{m,n} = c * sqrt((mπ/a)^2 + (nπ/b)^2).

    Args:
        icbc: {'u0': Expr, 'ut0': Expr(optional)} or u0 Expr
        geom: {'x_min':..., 'x_max':..., 'y_min':..., 'y_max':..., 't_min':...}
        phys: {'c':...}
        res: int or [Mx, My] (series truncation per dimension; if int, Mx=My=res)

    Returns:
        SymPy Expr u(x,y,t)
    """
    x_min = 0.0 if geom is None else float(geom.get("x_min", 0.0))
    x_max = 1.0 if geom is None else float(geom.get("x_max", 1.0))
    y_min = 0.0 if geom is None else float(geom.get("y_min", 0.0))
    y_max = 1.0 if geom is None else float(geom.get("y_max", 1.0))
    t_min = 0.0 if geom is None else float(geom.get("t_min", 0.0))

    a = x_max - x_min
    b = y_max - y_min

    c = 1.0 if phys is None else float(phys.get("c", 1.0))
    if isinstance(res, int):
        Mx = My = res
    else:
        Mx = int(res[0])
        My = int(res[1]) if len(res) > 1 else Mx

    if isinstance(icbc, Mapping):
        u0_expr = icbc.get("u0")
        ut0_expr = icbc.get("ut0", None)
    else:
        u0_expr, ut0_expr = icbc, None

    # Interior grids for DST-I
    xi = x_min + (np.arange(1, Mx + 1) * a) / (Mx + 1)
    yi = y_min + (np.arange(1, My + 1) * b) / (My + 1)

    f_fn = sp.lambdify((sp.Symbol("x"), sp.Symbol("y")), u0_expr, "numpy")
    g_fn = (lambda X, Y: np.zeros_like(X)) if ut0_expr is None else sp.lambdify(
        (sp.Symbol("x"), sp.Symbol("y")), ut0_expr, "numpy"
    )

    Xg, Yg = np.meshgrid(xi, yi, indexing="xy")  # shapes (Mx, My)
    f_samp = np.asarray(f_fn(Xg, Yg), dtype=float)
    g_samp = np.asarray(g_fn(Xg, Yg), dtype=float)

    # 2D DST-I with 'ortho' norm: apply separable transforms along axes
    F = dst(dst(f_samp, type=1, axis=0, norm="ortho"), type=1, axis=1, norm="ortho")
    G = dst(dst(g_samp, type=1, axis=0, norm="ortho"), type=1, axis=1, norm="ortho")
    scale = np.sqrt(4.0 / ((Mx + 1) * (My + 1)))  # product of 1D scales sqrt(2/(Mx+1)) * sqrt(2/(My+1))

    u_expr = sp.Integer(0)
    for m in range(1, Mx + 1):
        kx = m * sp.pi / a
        sx = sp.sin(m * sp.pi * (sp.Symbol("x") - x_min) / a)
        for n in range(1, My + 1):
            ky = n * sp.pi / b
            sy = sp.sin(n * sp.pi * (sp.Symbol("y") - y_min) / b)
            omega = c * sp.sqrt(kx * kx + ky * ky)

            A_mn = float(scale * F[m - 1, n - 1])
            B_mn = float(scale * G[m - 1, n - 1])

            u_expr += (A_mn * sp.cos(omega * (sp.Symbol("t") - t_min)) + (B_mn / omega) * sp.sin(omega * (sp.Symbol("t") - t_min))) * sx * sy

    return u_expr


def get_pde_residual(
    u_expr: sp.Expr,
    geom: Mapping[str, float] | None = None,
    phys: Mapping[str, float] | None = None,
    res: int | Sequence[int] | None = 100,
) -> dict:
    """
    Wave 2D PDE residual:
        r(x,y,t) = u_tt - c^2*(u_xx + u_yy) on (x_min,x_max) x (y_min,y_max) x (t_min,t_max).

    Args:
        u_expr: SymPy solution expression u(x,y,t).
        geom: {'x_min':..., 'x_max':..., 'y_min':..., 'y_max':..., 't_min':..., 't_max':...}
        phys: {'c':...}
        res: N or [Nx, Ny, Nt] (sampling for MSE)

    Returns:
        {'MSE':..., 'L2':..., 'Linf':...}
    """
    x_min = 0.0 if geom is None else float(geom.get("x_min", 0.0))
    x_max = 1.0 if geom is None else float(geom.get("x_max", 1.0))
    y_min = 0.0 if geom is None else float(geom.get("y_min", 0.0))
    y_max = 1.0 if geom is None else float(geom.get("y_max", 1.0))
    t_min = 0.0 if geom is None else float(geom.get("t_min", 0.0))
    t_max = 1.0 if geom is None else float(geom.get("t_max", 1.0))
    c = 1.0 if phys is None else float(phys.get("c", 1.0))

    if isinstance(res, int):
        Nx = Ny = Nt = res
    else:
        Nx = int(res[0])
        Ny = int(res[1]) if len(res) > 1 else Nx
        Nt = int(res[2]) if len(res) > 2 else Nx

    x_sym, y_sym, t_sym = sp.symbols("x y t", real=True)

    r_expr = sp.diff(u_expr, t_sym, 2) - (c**2) * (sp.diff(u_expr, x_sym, 2) + sp.diff(u_expr, y_sym, 2))
    rf = sp.lambdify((x_sym, y_sym, t_sym), r_expr, "numpy")

    xi = np.linspace(x_min, x_max, Nx)
    yi = np.linspace(y_min, y_max, Ny)
    ti = np.linspace(t_min, t_max, Nt)
    X, Y, T = np.meshgrid(xi, yi, ti, indexing="xy")  # shapes (Nx, Ny, Nt)
    R = rf(X, Y, T)

    dx = (x_max - x_min) / max(Nx - 1, 1)
    dy = (y_max - y_min) / max(Ny - 1, 1)
    dt = (t_max - t_min) / max(Nt - 1, 1)

    return {
        "MSE": float(np.mean(R**2)),
        "L2": float(np.sqrt(np.sum(R**2) * dx * dy * dt)),
        "Linf": float(np.max(np.abs(R))),
    }


def get_icbc_error(
    u_expr: sp.Expr,
    icbc: Mapping[str, sp.Expr] | sp.Expr,
    geom: Mapping[str, float] | None = None,
    res: int | Sequence[int] | None = 100,
) -> dict:
    """
    IC/BC MSE for wave 2D PDE
    IC at t = t_min:
        u(x,y,t_min)=u0(x,y), u_t(x,y,t_min)=ut0(x,y)
    BC (Dirichlet homogeneous):
        u(x_min,y,t)=0, u(x_max,y,t)=0, u(x,y_min,t)=0, u(x,y_max,t)=0

    Args:
        u_expr: SymPy solution expression u(x,y,t)
        icbc: {'u0':Expr, 'ut0':Expr(optional)} or u0: Expr
        geom: {'x_min':..., 'x_max':..., 'y_min':..., 'y_max':..., 't_min':..., 't_max':...}
        res: N or [Nx, Ny, Nt] (sampling)

    Returns:
        {'IC_u_MSE':..., 'IC_ut_MSE':..., 'BC_xmin_MSE':..., 'BC_xmax_MSE':..., 'BC_ymin_MSE':..., 'BC_ymax_MSE':..., 'MSE':...}
    """
    x_min = 0.0 if geom is None else float(geom.get("x_min", 0.0))
    x_max = 1.0 if geom is None else float(geom.get("x_max", 1.0))
    y_min = 0.0 if geom is None else float(geom.get("y_min", 0.0))
    y_max = 1.0 if geom is None else float(geom.get("y_max", 1.0))
    t_min = 0.0 if geom is None else float(geom.get("t_min", 0.0))
    t_max = 1.0 if geom is None else float(geom.get("t_max", 1.0))

    if isinstance(res, int):
        Nx = Ny = Nt = res
    else:
        Nx = int(res[0])
        Ny = int(res[1]) if len(res) > 1 else Nx
        Nt = int(res[2]) if len(res) > 2 else Nx

    if isinstance(icbc, Mapping):
        u0_expr = icbc.get("u0")
        ut0_expr = icbc.get("ut0", None)
    else:
        u0_expr, ut0_expr = icbc, None

    x_sym, y_sym, t_sym = sp.symbols("x y t", real=True)
    u_fn = sp.lambdify((x_sym, y_sym, t_sym), u_expr, "numpy")
    du_dt = sp.lambdify((x_sym, y_sym, t_sym), sp.diff(u_expr, t_sym), "numpy")
    f_fn = sp.lambdify((x_sym, y_sym), u0_expr, "numpy")
    g_fn = (lambda X, Y: np.zeros_like(X)) if ut0_expr is None else sp.lambdify((x_sym, y_sym), ut0_expr, "numpy")

    xi = np.linspace(x_min, x_max, Nx)
    yi = np.linspace(y_min, y_max, Ny)
    ti = np.linspace(t_min, t_max, Nt)

    # IC errors on (x,y) grid
    X2, Y2 = np.meshgrid(xi, yi, indexing="xy")
    ic_u_err = u_fn(X2, Y2, t_min) - f_fn(X2, Y2)
    ic_ut_err = du_dt(X2, Y2, t_min) - g_fn(X2, Y2)

    # BC errors on boundaries over time grid
    # x = x_min/x_max, y varies; and y = y_min/y_max, x varies
    Yt, Tt = np.meshgrid(yi, ti, indexing="xy")
    Xt, Tt2 = np.meshgrid(xi, ti, indexing="xy")
    bc_xmin_err = u_fn(x_min, Yt, Tt)      # target 0
    bc_xmax_err = u_fn(x_max, Yt, Tt)      # target 0
    bc_ymin_err = u_fn(Xt, y_min, Tt2)     # target 0
    bc_ymax_err = u_fn(Xt, y_max, Tt2)     # target 0

    n_ic = ic_u_err.size + ic_ut_err.size
    n_bc = bc_xmin_err.size + bc_xmax_err.size + bc_ymin_err.size + bc_ymax_err.size
    total = (np.sum(ic_u_err**2) + np.sum(ic_ut_err**2) +
             np.sum(bc_xmin_err**2) + np.sum(bc_xmax_err**2) +
             np.sum(bc_ymin_err**2) + np.sum(bc_ymax_err**2)) / (n_ic + n_bc)

    return {
        "IC_u_MSE": float(np.mean(ic_u_err**2)),
        "IC_ut_MSE": float(np.mean(ic_ut_err**2)),
        "BC_xmin_MSE": float(np.mean(bc_xmin_err**2)),
        "BC_xmax_MSE": float(np.mean(bc_xmax_err**2)),
        "BC_ymin_MSE": float(np.mean(bc_ymin_err**2)),
        "BC_ymax_MSE": float(np.mean(bc_ymax_err**2)),
        "MSE": float(total),
    }