import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from typing import Union
from pathlib import Path

from .model import LieSolver
from .dataloader import DataLoader


def sorted_idx(arr, dim, val, sort_dim):
    """ Indices where arr[:, dim] ≈ val, ordered by arr[:, sort_dim]"""
    mask = np.isclose(arr[:, dim], val)
    idx = np.where(mask)[0]
    return idx[np.argsort(arr[idx, sort_dim])]

def plot_ic_bc(
        model: LieSolver,
        data: DataLoader,
        decompose = False,
        mark_idx = [],
        filepath: Union[str, Path, None] = None
        ):
    """Plot initial and boundary condition curves with model predictions.
    Args:
       model (LieSolver | LieModule): Predictor providing outputs.
       data (DataLoader): Provides eval_x, eval_y, range_time, range_spatial.
       filepath (str | Path, optional): Path to save the figure.
    Returns:
       fig (Figure), ax (Axes or ndarray of Axes): Created figure and axes.
    """
    x = data.domain_x
    y_true = data.domain_y
    mark_idx = mark_idx if isinstance(mark_idx, list) else [mark_idx]
    y_pred = model(x)

    def plot_decomposition(ax, idx, axis):
        first_label_drawn = False
        for i, term in enumerate(model.terms):
            phi = term.base.eval(x[idx], term.params)
            comp = model.amplitudes[i] * phi
            alpha, color = (0.9, 'g') if i in mark_idx else (0.15, 'C0')       
            if not first_label_drawn:
                ax.plot(x[idx, axis], comp, color=color, alpha=alpha, label='terms')
                first_label_drawn = True
            else:
                ax.plot(x[idx, axis], comp, color=color, alpha=alpha)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    def ic_bc_subplot(ax_i, coord:str, ic_bc_val):
        coords = ['x', 't']
        ic_bc_dim = coords.index(coord)
        idx = sorted_idx(x, ic_bc_dim, ic_bc_val, 1-ic_bc_dim)
        ax[ax_i].plot(x[idx, 1-ic_bc_dim], y_true[idx], 'k', label="True")
        ax[ax_i].plot(x[idx, 1-ic_bc_dim], y_pred[idx], 'r', label="Pred")
        mse = np.mean((y_true[idx] - y_pred[idx])**2)
        ax[ax_i].set_title(f"{coords[ic_bc_dim]}={ic_bc_val} | MSE={mse:.1e}")
        ax[ax_i].set_xlabel(coords[1-ic_bc_dim])
        plot_decomposition(ax[ax_i], idx, 1-ic_bc_dim) if decompose else None
        ax[ax_i].legend()
    
    ic_bc_subplot(0, 't', data.bounds[1][0])
    ic_bc_subplot(1, 'x', data.bounds[0][0])
    ic_bc_subplot(2, 'x', data.bounds[0][1])

    fig.tight_layout()
    if filepath:
        fig.savefig(filepath, dpi=300)
    return fig, ax



def plot_2d_domain(
        model: LieSolver,
        data: DataLoader,
        filepath: Union[str, Path, None] = None
        ):
    """Plot predicted, true, and error fields over (x, t) with pcolormesh.
    Args:
       model (LieSolver | LieModule): Predictor providing outputs.
       data (DataLoader): Has eval_x (N,2) and eval_y (N,) on a grid.
       filename (str | Path, optional): Path to save the figure.
    Returns:
       fig (Figure), ax (Axes or ndarray of Axes): Created figure and axes.
    """
    x = data.domain_x
    nt = np.unique(x[:, 1]).size
    nx = x.shape[0] // nt
    X, T = x[:, 0].reshape(nt, nx), x[:, 1].reshape(nt, nx)

    y_true = data.domain_y.reshape(nt, nx)
    y_pred = model(x)

    y_pred = y_pred.reshape(nt, nx)
    y_err  = np.abs(y_true - y_pred)
    l2re = np.sqrt(np.sum(y_err**2) / np.sum(y_true**2))
    print(f"L2RE domain: {l2re:.2e}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for axi, z, title, cmap in zip(
        ax, 
        (y_pred, y_true, y_err), 
        ("Predicted", "True", f"Error | L2RE: {l2re:.2e}"), 
        ("magma",)*2+("binary",)
    ):
        pcm = axi.pcolormesh(X, T, z, shading="auto", cmap=cmap)
        fig.colorbar(pcm, ax=axi)
        axi.set_title(title)
        axi.set_xlabel("x")
        axi.set_ylabel("t") if axi is ax[0] else None   
    
    fig.tight_layout()
    if filepath:
        fig.savefig(filepath, dpi=300)
    return fig, ax

def plot_fit_history(state: "FitState", save_to=None):
    """Plot training set MSE vs number of parameters in model (log scale).
    Args:
       state (FitState): Provides nparams_hist, mse_hist
       save_to (str | Path, optional): Path to save the figure.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(state.nterms_hist, state.train_mse_hist, 'k', marker='.', 
             label=f'train MSE | final {state.train_mse_hist[-1]:.1e}')
    plt.plot(state.nterms_hist, state.test_mse_hist, 'g', marker='.', alpha=0.5,
             label=f'test MSE            {state.test_mse_hist[-1]:.1e}')
    plt.plot(state.nterms_hist, state.domain_mse_hist, 'b', marker='.', alpha=0.5,
             label=f'domain MSE      {state.domain_mse_hist[-1]:.1e}')

    domain_icbc_ratio = []
    for i in range(len(state.test_mse_hist)):
        domain_icbc_ratio.append(state.domain_mse_hist[i]/state.test_mse_hist[i])
        
    plt.plot(state.nterms_hist, domain_icbc_ratio, 'k--', 
            #  label=r'ratio $\frac{L2REdomain}{L2REtest}$', 
            #  label=r'ratio ${ L2RE_{domain} }/{ L2RE_{test} }$', 
             label=r'ratio MSE$_{domain}$ / MSE$_{test}$', 
             alpha=0.7)

    label_map = {nterms: fr"{nterms}$_{{{nparams}}}$" for nterms, nparams in zip(state.nterms_hist, state.nparams_hist)}
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: label_map.get(int(round(x)), "") if abs(x - round(x)) < 0.25 and int(round(x)) in label_map else ""
    ))
    print(f"nparams: {state.nparams_hist[-1]}")
    # plt.title('Fitting progress over number of added terms (parameters)')
    plt.xlabel(r"terms$_{{parameters}}$")
    # plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=300)
    else:
        plt.show()
    plt.close()

def plot_train_history(state: "TrainState", *, save_to=None):
    """TORCH RELATED 
    Plot training, test, and eval losses vs global iteration (log scale).
    Args:
       state (TrainState): Provides iter_hist, *_loss_hist, stage_iter_hist.
       save_to (str | Path, optional): Path to save the figure.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(state.iter_hist, state.train_loss_hist, label="train", marker='.')
    plt.plot(state.iter_hist, state.test_loss_hist, label="test", marker='.')
    plt.plot(state.iter_hist, state.eval_loss_hist, label="eval", marker='.')
    
    for stage_iter in state.stage_iter_hist[1:]:
        plt.axvline(stage_iter, color='black', linestyle='--', alpha=0.2)

    plt.xlabel("iteration")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=300)
    else:
        plt.show()
    plt.close()

