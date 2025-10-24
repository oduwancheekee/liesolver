# wave2d_kernels_exp.py
import sympy as sp
import torch
from model_new import LieModule, Transformation
from model import LieSymmetryNet

# ---- global symbols --------------------------------------------------------
x, y, t = sp.symbols('x y t', real=True)

# ---- kernels (transformations on u(x,y,t)) ---------------------------------
def shift_x(expr: sp.Expr, eps):
    return expr.subs(x, x - eps)

def shift_y(expr: sp.Expr, eps):
    return expr.subs(y, y - eps)

def shift_t(expr: sp.Expr, eps):
    return expr.subs(t, t + eps)

def rotation(expr: sp.Expr, eps):
    c = sp.cos(eps)
    s = sp.sin(eps)
    return expr.subs({
        x: c*x + s*y,
        y: c*y - s*x
    })

def hyperbolic_rotation_x(expr: sp.Expr, eps):
    ch = sp.cosh(eps)
    sh = sp.sinh(eps)
    return expr.subs({
        x:  ch*x - sh*t,
        t:  ch*t - sh*x
    })

def hyperbolic_rotation_y(expr: sp.Expr, eps):
    ch = sp.cosh(eps)
    sh = sp.sinh(eps)
    return expr.subs({
        y:  ch*y - sh*t,
        t:  ch*t - sh*y
    })

def dilatation(expr: sp.Expr, eps):
    s = sp.exp(eps)
    return expr.subs({
        x: s*x,
        y: s*y,
        t: s*t
    })

def inversion_x(expr: sp.Expr, eps):
    D = 1 + 2*eps*x - eps**2 * (t**2 - x**2 - y**2)
    mapped = expr.subs({
        x: (x - eps*(t**2 - x**2 - y**2)) / D,
        y:  y / D,
        t:  t / D,
    })
    return mapped / sp.sqrt(D)

def inversion_y(expr: sp.Expr, eps):
    D = 1 + 2*eps*y - eps**2 * (t**2 - x**2 - y**2)
    mapped = expr.subs({
        x:  x / D,
        y:  (y - eps*(t**2 - x**2 - y**2)) / D,
        t:  t / D,
    })
    return mapped / sp.sqrt(D)

def inversion_t(expr: sp.Expr, eps):
    D = 1 + 2*eps*t + eps**2 * (t**2 - x**2 - y**2)
    mapped = expr.subs({
        x:  x / D,
        y:  y / D,
        t:  (t + eps*(t**2 - x**2 - y**2)) / D,
    })
    return mapped / sp.sqrt(D)

def scale_u(expr: sp.Expr, eps):
    return sp.exp(eps) * expr

# ---- transformation registry ----------------------------------------------
wave2d_trafos: list[Transformation] = [
    Transformation(kernel=shift_x),
    Transformation(kernel=shift_y),
    Transformation(kernel=shift_t),
    Transformation(kernel=rotation),
    Transformation(kernel=hyperbolic_rotation_x),
    Transformation(kernel=hyperbolic_rotation_y),
    Transformation(kernel=dilatation),
    Transformation(kernel=inversion_x,
                   reparam=torch.nn.functional.softplus,
                   init_mean=-3., init_std=1.,
                   expensive=True),
    Transformation(kernel=inversion_y,
                   reparam=torch.nn.functional.softplus,
                   init_mean=-3., init_std=1.,
                   expensive=True),
    Transformation(kernel=inversion_t,
                   reparam=torch.nn.functional.softplus,
                   init_mean=-3., init_std=1.,
                   expensive=True),
    Transformation(kernel=scale_u),
]

# ---- base solutions --------------------------------------------------------
def base_solution():
    # Exact solution of u_tt - u_xx - u_yy = 0:
    # u(x,y,t) = sin(x) sin(y) cos(sqrt(2) * t)
    return sp.sin(x) * sp.sin(y) * sp.cos(sp.sqrt(2) * t)

def base_solution2():
    expr = sp.Add(
        1,
        sp.Mul(0, x, evaluate=False),
        sp.Mul(0, y, evaluate=False),
        sp.Mul(0, t, evaluate=False),
        evaluate=False
    )
    return expr

# ---- PDE residual for 2D wave equation ------------------------------------
def pde_residual(model: LieModule,
                 n_samples=5_000,
                 x_min=0.0, x_max=1.0,
                 y_min=0.0, y_max=1.0,
                 t_min=0.0,  t_max=1.0):
    """
    Return mean‑squared residual of u_tt - u_xx - u_yy on random samples.
    """
    # sample (x,y,t)
    x_s = torch.empty(n_samples, 1).uniform_(x_min, x_max).requires_grad_(True)
    y_s = torch.empty(n_samples, 1).uniform_(y_min, y_max).requires_grad_(True)
    t_s = torch.empty(n_samples, 1).uniform_(t_min, t_max).requires_grad_(True)

    # forward pass
    u = model(torch.concat((x_s, y_s, t_s), dim=1))  # (n_samples,)

    # first derivatives
    ones = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x_s, ones, create_graph=True)[0].squeeze()
    u_y = torch.autograd.grad(u, y_s, ones, create_graph=True)[0].squeeze()
    u_t = torch.autograd.grad(u, t_s, ones, create_graph=True)[0].squeeze()

    # second derivatives
    u_xx = torch.autograd.grad(u_x, x_s, ones, create_graph=True)[0].squeeze()
    u_yy = torch.autograd.grad(u_y, y_s, ones, create_graph=True)[0].squeeze()
    u_tt = torch.autograd.grad(u_t, t_s, ones, create_graph=True)[0].squeeze()

    # residual
    R = u_tt - u_xx - u_yy
    return torch.mean(R**2).item()