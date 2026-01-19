import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from typing import Union, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from .model import LieSolver
from .dataloader import DataLoader


class MetricPlotter(ABC):
    """Base class for plotting tracked metrics.
    
    Users can inherit from this class to define custom visualizations
    for metrics tracked during training.
    
    Example:
        class ConditionNumberPlotter(MetricPlotter):
            def plot(self, fit_state, metric_name='condition_number', 
                    filepath=None, **kwargs):
                values = fit_state.custom_metrics[metric_name]
                steps = fit_state.nterms_hist
                
                plt.figure(figsize=(8, 5))
                plt.plot(steps, values, marker='o')
                plt.xlabel('Number of Terms')
                plt.ylabel('Condition Number')
                plt.yscale('log')
                plt.title('Condition Number Evolution')
                
                if filepath:
                    plt.savefig(filepath, dpi=300)
                plt.close()
    """
    
    @abstractmethod
    def plot(self, fit_state, metric_name: str, filepath: Optional[Union[str, Path]] = None, **kwargs):
        """Plot metric values from fit_state.
        
        Args:
            fit_state (FitState): FitState instance with tracked metrics.
            metric_name (str): Name of the metric to plot.
            filepath (Optional[Union[str, Path]]): Path to save the figure.
            **kwargs: Additional plotting parameters.
        """
        raise NotImplementedError("Subclasses must implement plot()")


class SimpleMetricPlotter(MetricPlotter):
    """Simple line plot for any tracked metric."""
    
    def plot(self, fit_state, metric_name: str, 
            filepath: Optional[Union[str, Path]] = None,
            figsize=(8, 5), 
            ylabel: Optional[str] = None,
            title: Optional[str] = None,
            log_scale: bool = False,
            **kwargs):
        """Plot metric values as a simple line plot.
        
        Args:
            fit_state (FitState): FitState instance with tracked metrics.
            metric_name (str): Name of the metric to plot.
            filepath (Optional[Union[str, Path]]): Path to save the figure.
            figsize (tuple): Figure size. Default (8, 5).
            ylabel (Optional[str]): Y-axis label. Defaults to metric_name.
            title (Optional[str]): Plot title. Defaults to metric_name evolution.
            log_scale (bool): Use log scale for y-axis. Default False.
            **kwargs: Additional parameters passed to plt.plot().
        """
        if metric_name not in fit_state.metrics_history:
            raise ValueError(f"Metric '{metric_name}' not found in fit_state.metrics_history")
        
        values = fit_state.metrics_history[metric_name]
        steps = fit_state.nterms_hist[:len(values)]
        
        plt.figure(figsize=figsize)
        plt.plot(steps, values, marker='o', **kwargs)
        plt.xlabel(r"N$_{\text{terms}}$")
        plt.ylabel(ylabel or metric_name)
        if log_scale:
            plt.yscale('log')
        plt.title(title or f'{metric_name} evolution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300)
        else:
            plt.show()
        plt.close()


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
       model (LieSolver): Predictor providing outputs.
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
        for i, brick in enumerate(model.bricks):
            phi = brick.family.eval(x[idx], brick.params)
            comp = model.amplitudes[i] * phi
            alpha, color = (0.9, 'g') if i in mark_idx else (0.15, 'C0')       
            if not first_label_drawn:
                ax.plot(x[idx, axis], comp, color=color, alpha=alpha, label='bricks')
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

def plot_fit_history(state, save_to=None):
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
             label=r'ratio MSE$_{\text{domain}}$ / MSE$_{\text{test}}$', 
             alpha=0.7)

    label_map = {nterms: fr"{nterms}({nparams})" for nterms, nparams in zip(state.nterms_hist, state.nparams_hist)}
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: label_map.get(int(round(x)), "") if abs(x - round(x)) < 0.25 and int(round(x)) in label_map else ""
    ))
    ax.yaxis.grid(True, which='major', linestyle='--', color='k', linewidth=0.5, alpha=0.25)

    print(f"nparams: {state.nparams_hist[-1]}")
    # plt.title('Fitting progress over number of added terms (parameters)')
    plt.xlabel(r"N$_{\text{terms}}$(N$_{\text{parameters}}$)")
    # plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_amplitude_evolution(fit_state, figsize=(12, 6), margin_fraction=0.5, filepath: Union[str, Path, None] = None):
    """Plot amplitude evolution across fitting history with relative amplitudes.
    
    Visualizes how each term's amplitude (normalized by total) evolves throughout
    the fitting process. Each term is shown at a fixed x-position, with points
    scattered horizontally for visibility. Transparency indicates age (older = more
    transparent), color indicates final state (red = final, black = history).
    
    Args:
        fit_state (FitState): FitState instance from trainer.
        figsize (tuple): Figure size as (width, height). Default (12, 6).
        margin_fraction (float): Margin as fraction of y-range from percentiles.
                                Default 0.5 (50% on each side).
        filepath (str | Path, optional): Path to save the figure.
    """
    from matplotlib.lines import Line2D
    
    history_amplitudes = fit_state.amplitudes_hist
    history_steps = fit_state.step_type_hist
    
    if not history_amplitudes:
        print("No history available")
        return
    
    # Find max number of terms across all steps
    max_terms = max(len(a) for a in history_amplitudes)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all y_values for statistics
    all_y_values = []
    
    # For each term (amplitude position)
    for term_idx in range(max_terms):
        x_positions = []
        y_values = []
        alphas = []
        
        # For each step in history
        for step_idx, amplitudes in enumerate(history_amplitudes):
            # Only plot if this term exists at this step
            if term_idx < len(amplitudes):
                x_positions.append(term_idx)
                # Normalize amplitude by sum of absolute amplitudes at this step
                amplitude_sum = np.sum(np.abs(amplitudes))
                normalized_amp = amplitudes[term_idx] / amplitude_sum if amplitude_sum > 0 else 0
                y_values.append(normalized_amp)
                all_y_values.append(normalized_amp)
                # Transparency increases with step (older = more transparent)
                alphas.append(step_idx / len(history_amplitudes) / 2)
        alphas[-1] = 1.0
        
        # Plot all points for this term with varying transparency
        for i, (x, y, alpha) in enumerate(zip(x_positions, y_values, alphas)):
            # Final point (current model) in red, others in black
            is_final = (i == len(x_positions) - 1)
            
            color = 'red' if is_final else 'black'
            ax.scatter(x - 0.33 + (i+1)/len(x_positions)/3, y, s=50, alpha=alpha, 
                      color=color, edgecolors='none')
    
    # Calculate ylim based on percentiles to exclude outliers
    all_y_values = np.array(all_y_values)
    q5, q95 = np.percentile(all_y_values, [5, 95])
    y_range = q95 - q5
    margin = margin_fraction * y_range
    y_min = q5 - margin
    y_max = q95 + margin
    ax.set_ylim(y_min, y_max)
    
    # Set ticks at all integer positions for grid lines
    all_ticks = range(max_terms)
    ax.set_xticks(all_ticks)
    
    # Determine which ticks to label to avoid overlap
    step = max(1, max_terms // 10)
    ax.set_xticklabels([str(i) if i % step == 0 else '' for i in all_ticks])
    
    ax.grid(True, axis='x', alpha=0.3)
    ax.axhline(y=0, alpha=0.3, color='black', linewidth=0.3)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='history'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.3, markersize=8, label='older history'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='final')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_xlabel('Term Index (Amplitude Order)', fontsize=12)
    ax.set_ylabel('Relative Amplitude (fraction of total)', fontsize=12)
    ax.set_title('Relative Amplitude Evolution Across Fitting History', fontsize=13)
    
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_custom_metric(fit_state, metric_name: str, 
                      filepath: Optional[Union[str, Path]] = None,
                      plotter: Optional[MetricPlotter] = None,
                      **kwargs):
    """Convenience function to plot a custom metric.
    
    Args:
        fit_state (FitState): FitState instance with tracked metrics.
        metric_name (str): Name of the metric to plot.
        filepath (Optional[Union[str, Path]]): Path to save the figure.
        plotter (Optional[MetricPlotter]): Custom plotter instance. 
                                          Defaults to SimpleMetricPlotter.
        **kwargs: Additional parameters passed to the plotter.
    """
    if plotter is None:
        plotter = SimpleMetricPlotter()
    
    plotter.plot(fit_state, metric_name, filepath=filepath, **kwargs)
