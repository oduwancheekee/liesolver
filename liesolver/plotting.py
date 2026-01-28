import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from typing import Union, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from .model import LieSolver
from .dataloader import DataLoader


# Colorblind-friendly palette (Wong, 2011)
COLORS = {
    'true': '#000000',        # Black - ground truth
    'pred': '#E69F00',        # Orange - predictions (High contrast vs Black)
    'error': '#D55E00',       # Vermillion - errors
    'bricks': '#56B4E9',      # Sky blue - brick decomposition
    'train': '#000000',       # Black - train MSE
    'test': '#E69F00',        # Orange - test MSE (High contrast vs Black)
    'domain': '#009E73',      # Bluish Green - domain MSE (Distinct from Black/Orange)
    'ratio': '#CC79A7',       # Reddish purple - ratio line
}

# Figure sizing presets
SIZE_PRESETS = {
    'default': {  # For 16cm width
        'ic_bc_figsize': (15, 5),
        'domain_figsize': (12, 5),
        'history_figsize': (6, 4),
        'title_fontsize': 11,
        'label_fontsize': 10,
        'tick_fontsize': 9,
        'legend_fontsize': 9,
        'linewidth': 2.0,
        'markersize': 5,
    },
    'compact': {  # For 8cm width
        'ic_bc_figsize': (7.5, 2.8),
        'domain_figsize': (7, 3),
        'history_figsize': (4.5, 3.5),
        'title_fontsize': 14,
        'label_fontsize': 13,
        'tick_fontsize': 12,
        'legend_fontsize': 10,
        'linewidth': 1.4,
        'markersize': 4,
    },
}


class MetricPlotter(ABC):
    """Base class for plotting tracked metrics.
    
    Users can inherit from this class to define custom visualizations
    for metrics tracked during training.
    
    Example:
        class ConditionNumberPlotter(MetricPlotter):
            def plot(self, fit_state, metric_name='condition_number', 
                    filepath=None, **kwargs):
                values = fit_state.custom_metrics[metric_name]
                steps = fit_state.nbricks_hist
                
                plt.figure(figsize=(8, 5))
                plt.plot(steps, values, marker='o')
                plt.xlabel('Number of Bricks')
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
        steps = fit_state.nbricks_hist[:len(values)]
        
        plt.figure(figsize=figsize)
        plt.plot(steps, values, marker='o', **kwargs)
        plt.xlabel(r"N$_{\text{bricks}}$")
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
        decompose: bool = False,
        mark_idx = [],
        compact: bool = False,
        filepath: Union[str, Path, None] = None
        ):
    """Plot initial and boundary condition curves with model predictions.
    Args:
       model (LieSolver): Predictor providing outputs.
       data (DataLoader): Provides eval_x, eval_y, range_time, range_spatial.
       compact (bool): If True, use smaller figure with larger relative fonts.
       filepath (str | Path, optional): Path to save the figure.
    Returns:
       fig (Figure), ax (Axes or ndarray of Axes): Created figure and axes.
    """
    sizes = SIZE_PRESETS['compact'] if compact else SIZE_PRESETS['default']
    
    x = data.domain_x
    y_true = data.domain_y
    mark_idx = mark_idx if isinstance(mark_idx, list) else [mark_idx]
    y_pred = model(x)

    def plot_decomposition(ax, idx, axis, add_label=False):
        for i, brick in enumerate(model.bricks):
            phi = brick.family.eval(x[idx], brick.params)
            comp = model.amplitudes[i] * phi
            alpha = 0.9 if i in mark_idx else 0.2
            color = COLORS['error'] if i in mark_idx else COLORS['bricks']
            label = 'Bricks' if (add_label and i == 0) else None
            ax.plot(x[idx, axis], comp, color=color, alpha=alpha, 
                   linewidth=sizes['linewidth']*0.8, label=label)

    fig, ax = plt.subplots(1, 3, figsize=sizes['ic_bc_figsize'])
    
    # Track handles for shared legend
    legend_handles = []
    legend_labels = []

    def ic_bc_subplot(ax_i, coord: str, ic_bc_val, add_legend_entries=False):
        coords = ['x', 't']
        ic_bc_dim = coords.index(coord)
        idx = sorted_idx(x, ic_bc_dim, ic_bc_val, 1-ic_bc_dim)
        if decompose:
            plot_decomposition(ax[ax_i], idx, 1-ic_bc_dim, add_label=add_legend_entries)
        
        line_true, = ax[ax_i].plot(x[idx, 1-ic_bc_dim], y_true[idx], 
                                   color=COLORS['true'], linewidth=sizes['linewidth'],
                                   linestyle='-')
        

        line_pred, = ax[ax_i].plot(x[idx, 1-ic_bc_dim], y_pred[idx], 
                                   color=COLORS['pred'], linewidth=sizes['linewidth'],
                                   linestyle='--')
        
        if add_legend_entries:
            legend_handles.extend([line_true, line_pred])
            legend_labels.extend(['True', 'Predicted'])
        
        mse = np.mean((y_true[idx] - y_pred[idx])**2)
        # ax[ax_i].set_title(f"{coords[ic_bc_dim]}={ic_bc_val} | MSE={mse:.1e}",
        #                   fontsize=sizes['title_fontsize'])
        if ax_i == 0:
            title = f"IC (MSE: {np.format_float_scientific(mse, precision=1, exp_digits=1, trim='-')})"
        elif ax_i == 1:
            title = f"BC left (MSE: {np.format_float_scientific(mse, precision=1, exp_digits=1, trim='-')})"
        else:  # ax_i == 2
            title = f"BC right (MSE: {np.format_float_scientific(mse, precision=1, exp_digits=1, trim='-')})"
        ax[ax_i].set_title(title, fontsize=sizes['title_fontsize'])
        ax[ax_i].set_xlabel(coords[1-ic_bc_dim], fontsize=sizes['label_fontsize'])
        ax[ax_i].tick_params(labelsize=sizes['tick_fontsize'])

    
    ic_bc_subplot(0, 't', data.bounds[1][0], add_legend_entries=True)
    ic_bc_subplot(1, 'x', data.bounds[0][0])
    ic_bc_subplot(2, 'x', data.bounds[0][1])
    
    # Add bricks to legend if decompose mode
    if decompose:
        from matplotlib.lines import Line2D
        brick_line = Line2D([0], [0], color=COLORS['bricks'], alpha=0.4,
                           linewidth=sizes['linewidth']*0.8)
        legend_handles.append(brick_line)
        legend_labels.append('Weighted\nbricks')
    
    # Single shared legend on the right side
    fig.legend(legend_handles, legend_labels, 
              loc='center left', fontsize=sizes['legend_fontsize'],
              framealpha=0.9, bbox_to_anchor=(.93, 0.72))
    
    fig.tight_layout(rect=[0, 0, 0.96, 1])  # Make room for legend on right
    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
    return fig, ax



def plot_2d_domain(
        model: LieSolver,
        data: DataLoader,
        compact: bool = False,
        filepath: Union[str, Path, None] = None
        ):
    """Plot predicted and error fields over (x, t) with pcolormesh.
    Args:
       model (LieSolver | LieModule): Predictor providing outputs.
       data (DataLoader): Has eval_x (N,2) and eval_y (N,) on a grid.
       compact (bool): If True, use smaller figure with larger relative fonts.
       filepath (str | Path, optional): Path to save the figure.
    Returns:
       fig (Figure), ax (Axes or ndarray of Axes): Created figure and axes.
    """
    sizes = SIZE_PRESETS['compact'] if compact else SIZE_PRESETS['default']
    
    x = data.domain_x
    nt = np.unique(x[:, 1]).size
    nx = x.shape[0] // nt
    X, T = x[:, 0].reshape(nt, nx), x[:, 1].reshape(nt, nx)

    y_true = data.domain_y.reshape(nt, nx)
    y_pred = model(x).reshape(nt, nx)
    y_err = np.abs(y_true - y_pred)
    l2re = np.sqrt(np.sum(y_err**2) / np.sum(y_true**2))
    print(f"L2RE domain: {l2re:.1e}")
    
    fig, ax = plt.subplots(1, 2, figsize=sizes['domain_figsize'])
    
    # Prediction plot
    pcm_pred = ax[0].pcolormesh(X, T, y_pred, shading="auto", cmap="viridis")
    fig.colorbar(pcm_pred, ax=ax[0])
    ax[0].set_title("Predicted", fontsize=sizes['title_fontsize'])
    ax[0].set_xlabel("x", fontsize=sizes['label_fontsize'])
    ax[0].set_ylabel("t", fontsize=sizes['label_fontsize'])
    ax[0].tick_params(labelsize=sizes['tick_fontsize'])
    ax[0].set_aspect('auto')
    
    # Error plot - use grayscale for magnitude (white=0, black=max)
    pcm_err = ax[1].pcolormesh(X, T, y_err, shading="auto", cmap="gray_r")
    cbar_err = fig.colorbar(pcm_err, ax=ax[1])
    cbar_err.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1e}'))
    ax[1].set_title(f"Error (L2RE: {l2re:.1e})", fontsize=sizes['title_fontsize'])
    ax[1].set_xlabel("x", fontsize=sizes['label_fontsize'])
    ax[1].tick_params(labelsize=sizes['tick_fontsize'])
    ax[1].set_aspect('auto')
    ax[1].set_yticklabels([])  # Hide y-ticks on error plot

    fig.tight_layout()
    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_fit_history(state, compact: bool = False, save_to=None):
    """Plot training set MSE vs number of parameters in model (log scale).
    Args:
       state (FitState): Provides nparams_hist, mse_hist
       compact (bool): If True, use smaller figure with larger relative fonts.
       save_to (str | Path, optional): Path to save the figure.
    Returns:
       fig (Figure), ax (Axes): Created figure and axes.
    """
    sizes = SIZE_PRESETS['compact'] if compact else SIZE_PRESETS['default']
    
    fig, ax = plt.subplots(figsize=sizes['history_figsize'])
    
    ax.plot(state.nbricks_hist, state.train_mse_hist, 
            color=COLORS['train'], marker='', markersize=sizes['markersize'],
            linewidth=1.3*sizes['linewidth'], linestyle='-',
            label=f'train | final {state.train_mse_hist[-1]:.1e}')
    ax.plot(state.nbricks_hist, state.test_mse_hist, 
            color=COLORS['test'], marker='', markersize=sizes['markersize'],
            linewidth=1.3*sizes['linewidth'], linestyle='--', alpha=0.8,
            label=f'test            {state.test_mse_hist[-1]:.1e}')
    ax.plot(state.nbricks_hist, state.domain_mse_hist, 
            color=COLORS['domain'], marker='', markersize=sizes['markersize'],
            linewidth=sizes['linewidth'], linestyle='-.', alpha=0.8,
            label=f'domain      {state.domain_mse_hist[-1]:.1e}')

    domain_icbc_ratio = [state.domain_mse_hist[i]/state.test_mse_hist[i] 
                         for i in range(len(state.test_mse_hist))]
        
    ax.plot(state.nbricks_hist, domain_icbc_ratio, 
            color=COLORS['ratio'], linestyle='-', linewidth=sizes['linewidth'],
            label=r'ratio domain/test', 
            alpha=0.95)
    
    # vertical lines at refinement steps
    for i, step_type in enumerate(state.step_type_hist):
        if step_type == 'refine_all':
            ax.axvline(x=state.nbricks_hist[i], color='k', linestyle='-', linewidth=1, alpha=.5)
        elif step_type == 'refine_batch':
            ax.axvline(x=state.nbricks_hist[i], color='grey', linestyle='-', linewidth=1, alpha=.5)

    # Bottom x-axis: N_bricks
    bricks_label_map = {nbricks: fr"{nbricks}" 
                        for nbricks in state.nbricks_hist}
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: bricks_label_map.get(int(round(x)), "") if abs(x - round(x)) < 0.25 and int(round(x)) in bricks_label_map else ""
    ))
    ax.yaxis.grid(True, which='major', linestyle='--', color='k', linewidth=0.5, alpha=0.2)
    
    # Top x-axis: N_parameters  
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())  # Same x range as bottom axis
    params_label_map = {nbricks: fr"{nparams}" 
                        for nbricks, nparams in zip(state.nbricks_hist, state.nparams_hist)}
    ax2.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: params_label_map.get(int(round(x)), "") if abs(x - round(x)) < 0.25 and int(round(x)) in params_label_map else ""
    ))

    print(f"nparams: {state.nparams_hist[-1]}")
    ax.set_xlabel("# bricks", fontsize=sizes['label_fontsize'])
    ax2.set_xlabel("# parameters", fontsize=sizes['label_fontsize']/1.2)
    ax.set_ylabel("MSE", fontsize=sizes['label_fontsize'])
    ax.tick_params(labelsize=sizes['tick_fontsize'])
    ax.set_yscale("log")
    # if compact:
    #     ax.legend(fontsize=sizes['legend_fontsize'], loc=[0.45, 0.45])
    # else:
    ax.legend(fontsize=sizes['legend_fontsize'])

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_amplitude_evolution(fit_state, figsize=(12, 6), margin_fraction=0.5, filepath: Union[str, Path, None] = None):
    """Plot amplitude evolution across fitting history with relative amplitudes.
    
    Visualizes how each brick's amplitude (normalized by total) evolves throughout
    the fitting process. Each brick is shown at a fixed x-position, with points
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
    
    # Find max number of bricks across all steps
    max_bricks = max(len(a) for a in history_amplitudes)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all y_values for statistics
    all_y_values = []
    
    # For each brick (amplitude position)
    for brick_idx in range(max_bricks):
        x_positions = []
        y_values = []
        alphas = []
        
        # For each step in history
        for step_idx, amplitudes in enumerate(history_amplitudes):
            # Only plot if this brick exists at this step
            if brick_idx < len(amplitudes):
                x_positions.append(brick_idx)
                # Normalize amplitude by sum of absolute amplitudes at this step
                amplitude_sum = np.sum(np.abs(amplitudes))
                normalized_amp = amplitudes[brick_idx] / amplitude_sum if amplitude_sum > 0 else 0
                y_values.append(normalized_amp)
                all_y_values.append(normalized_amp)
                # Transparency increases with step (older = more transparent)
                alphas.append(step_idx / len(history_amplitudes) / 2)
        alphas[-1] = 1.0
        
        # Plot all points for this brick with varying transparency
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
    all_ticks = range(max_bricks)
    ax.set_xticks(all_ticks)
    
    # Determine which ticks to label to avoid overlap
    step = max(1, max_bricks // 10)
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
    
    ax.set_xlabel('Brick Index (Amplitude Order)', fontsize=12)
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
