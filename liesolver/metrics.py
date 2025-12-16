"""Metrics and analysis functions for LieSolver models."""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .model import LieSolver


def record_history(model: LieSolver, step_type: str) -> None:
    """Record current model state to history.
    
    Args:
        model (LieSolver): Model instance to record.
        step_type (str): Type of step ('add', 'refine', 'remove', 'add_defined_term', etc.).
    """
    model.history['amplitudes'].append(model.amplitudes.copy())
    # Store a snapshot of all term parameters
    term_params_snapshot = [term.params.copy() for term in model.terms]
    model.history['term_params'].append(term_params_snapshot)
    model.history['mse'].append(model.mse)
    model.history['n_terms'].append(len(model.terms))
    model.history['step_type'].append(step_type)


def condition_number(model: LieSolver, F: np.ndarray = None) -> float:
    """
    Compute the condition number of F^T F (ratio of largest to smallest eigenvalue).
    
    A low condition number (close to 1) indicates a well-conditioned system.
    A high condition number indicates numerical instability and potential ill-conditioning.
    
    Args:
        model (LieSolver): Model instance.
        F (Optional[np.ndarray]): Feature matrix. If None, uses current feature matrix.
    
    Returns:
        float: Condition number κ(F^T F) = λ_max / λ_min.
    """
    if F is None:
        F = model.feature_matrix(model.terms)
    
    if F.shape[1] == 0:
        return np.inf
    
    FtF = F.T @ F
    eigvals = np.linalg.eigvalsh(FtF)
    eigvals = np.abs(eigvals[eigvals > 1e-15])  # Filter near-zero eigenvalues
    
    if len(eigvals) == 0:
        return np.inf
    
    return float(np.max(eigvals) / np.min(eigvals))


def matrix_rank(model: LieSolver, F: np.ndarray = None, tol: float = 1e-10) -> int:
    """
    Compute the numerical rank of the feature matrix F.
    
    A full rank matrix has rank equal to min(L, M).
    Rank deficiency indicates linear dependence among base functions.
    
    Args:
        model (LieSolver): Model instance.
        F (Optional[np.ndarray]): Feature matrix. If None, uses current feature matrix.
        tol (float): Threshold for considering singular values as non-zero.
    
    Returns:
        int: Numerical rank of F.
    """
    if F is None:
        F = model.feature_matrix(model.terms)
    
    if F.shape[1] == 0:
        return 0
    
    _, s, _ = np.linalg.svd(F, full_matrices=False)
    return int(np.sum(s > tol))


def eigenvalue_distribution(model: LieSolver, F: np.ndarray = None) -> Tuple[np.ndarray, dict]:
    """
    Compute eigenvalues of F^T F and statistics on their distribution.
    
    For a well-conditioned problem with orthogonal bases, eigenvalues should be
    clustered and relatively uniform. High variance in eigenvalues indicates
    some bases are much more important than others.
    
    Args:
        model (LieSolver): Model instance.
        F (Optional[np.ndarray]): Feature matrix. If None, uses current feature matrix.
    
    Returns:
        Tuple[np.ndarray, dict]: 
            - eigenvalues (sorted descending)
            - stats dict with keys:
                - 'mean': mean eigenvalue
                - 'std': standard deviation of eigenvalues
                - 'min': minimum eigenvalue
                - 'max': maximum eigenvalue
                - 'ratio_max_min': ratio of max to min (condition number)
    """
    if F is None:
        F = model.feature_matrix(model.terms)
    
    if F.shape[1] == 0:
        return np.array([]), {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'ratio_max_min': np.inf}
    
    FtF = F.T @ F
    eigvals = np.linalg.eigvalsh(FtF)
    eigvals = np.sort(eigvals)[::-1]  # Sort descending
    eigvals = eigvals[eigvals > 1e-15]  # Filter near-zero eigenvalues
    
    if len(eigvals) == 0:
        return np.array([]), {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'ratio_max_min': np.inf}
    
    stats = {
        'mean': float(np.mean(eigvals)),
        'std': float(np.std(eigvals)),
        'min': float(np.min(eigvals)),
        'max': float(np.max(eigvals)),
        'ratio_max_min': float(np.max(eigvals) / np.min(eigvals))
    }
    
    return eigvals, stats


def base_orthogonality(model: LieSolver, X: np.ndarray = None) -> Tuple[float, dict]:
    """
    Check orthogonality of base functions using Hilbert space inner product.
    
    For a domain [a, b], the Hilbert space inner product is:
        <f_i, f_j> ≈ Σ f_i(x_k) * f_j(x_k) * dx  (quadrature approximation)
    
    The normalized metric is:
        max_{i ≠ j} |<f_i, f_j>| / (||f_i|| * ||f_j||)
    
    A value close to 0 indicates near-orthogonal bases (good).
    A value close to 1 indicates significant overlap (problematic).
    
    Args:
        model (LieSolver): Model instance.
        X (Optional[np.ndarray]): Data points for evaluating orthogonality.
                                 If None, uses training data.
    
    Returns:
        Tuple[float, dict]:
            - max_overlap: maximum normalized inner product between distinct bases
            - stats dict with keys:
                - 'mean_overlap': mean of all pairwise normalized overlaps
                - 'min_overlap': minimum normalized overlap
                - 'max_overlap': maximum normalized overlap
                - 'gram_matrix': full Gram matrix (M x M) of normalized overlaps
    """
    if X is None:
        X = model.X
    
    if len(model.terms) == 0:
        return 0.0, {'mean_overlap': 0.0, 'min_overlap': 0.0, 'max_overlap': 0.0, 'gram_matrix': np.array([])}
    
    M = len(model.terms)
    
    # Compute base function evaluations
    base_evals = []
    for term in model.terms:
        phi = term.base.eval(X, term.params)
        base_evals.append(phi)
    base_evals = np.array(base_evals)  # (M, L)
    
    # Compute norms (L2 norm over data points)
    norms = np.linalg.norm(base_evals, axis=1)  # (M,)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    
    # Compute Gram matrix (unnormalized inner products)
    gram_raw = base_evals @ base_evals.T  # (M, M)
    
    # Normalize by outer product of norms
    norm_outer = norms[:, None] * norms[None, :]
    gram_normalized = np.abs(gram_raw) / norm_outer
    
    # Zero out diagonal for max computation (we only care about i ≠ j)
    np.fill_diagonal(gram_normalized, 0.0)
    
    max_overlap = float(np.max(gram_normalized)) if gram_normalized.size > 0 else 0.0
    
    # Compute statistics on off-diagonal elements
    off_diag_indices = np.triu_indices(M, k=1)
    off_diag_values = gram_normalized[off_diag_indices]
    
    stats = {
        'mean_overlap': float(np.mean(off_diag_values)) if len(off_diag_values) > 0 else 0.0,
        'min_overlap': float(np.min(off_diag_values)) if len(off_diag_values) > 0 else 0.0,
        'max_overlap': max_overlap,
        'gram_matrix': gram_normalized,
    }
    
    return max_overlap, stats


def get_history_summary(model: LieSolver) -> dict:
    """Get summary statistics of the fitting history.
    
    Args:
        model (LieSolver): Model instance.
    
    Returns:
        dict: Summary with keys:
            - 'n_steps': total number of steps recorded
            - 'step_types': list of step types
            - 'mse_initial': MSE at start
            - 'mse_final': MSE at end
            - 'mse_improvement': percentage improvement
            - 'n_terms_added': number of terms added
            - 'n_refines': number of refinement steps
            - 'n_removals': number of terms removed
    """
    if not model.history['mse']:
        return {
            'n_steps': 0,
            'step_types': [],
            'mse_initial': np.nan,
            'mse_final': np.nan,
            'mse_improvement': 0.0,
            'n_terms_added': 0,
            'n_refines': 0,
            'n_removals': 0,
        }
    
    mse_arr = np.array(model.history['mse'])
    step_types = model.history['step_type']
    
    mse_initial = mse_arr[0] if len(mse_arr) > 0 else np.nan
    mse_final = mse_arr[-1] if len(mse_arr) > 0 else np.nan
    mse_improvement = 100.0 * (mse_initial - mse_final) / (mse_initial + 1e-12) if mse_initial > 0 else 0.0
    
    return {
        'n_steps': len(model.history['mse']),
        'step_types': step_types,
        'mse_initial': float(mse_initial),
        'mse_final': float(mse_final),
        'mse_improvement': float(mse_improvement),
        'n_terms_added': sum(1 for st in step_types if 'add' in st),
        'n_refines': sum(1 for st in step_types if 'refine' in st),
        'n_removals': sum(1 for st in step_types if 'remove' in st),
    }


def get_step_info(model: LieSolver, step_idx: int) -> dict:
    """Get detailed information about a specific step.
    
    Args:
        model (LieSolver): Model instance.
        step_idx (int): Index of the step in history.
    
    Returns:
        dict: Information with keys:
            - 'step_type': type of step
            - 'mse': MSE at this step
            - 'n_terms': number of terms at this step
            - 'amplitudes': amplitude vector
            - 'term_params': parameters of all terms
            - 'amplitude_norms': L2 norms of amplitudes
    """
    if step_idx < 0 or step_idx >= len(model.history['mse']):
        raise IndexError(f"step_idx {step_idx} out of range [0, {len(model.history['mse'])-1}]")
    
    amplitudes = model.history['amplitudes'][step_idx]
    term_params = model.history['term_params'][step_idx]
    
    return {
        'step_type': model.history['step_type'][step_idx],
        'mse': float(model.history['mse'][step_idx]),
        'n_terms': int(model.history['n_terms'][step_idx]),
        'amplitudes': amplitudes,
        'term_params': term_params,
        'amplitude_norms': float(np.linalg.norm(amplitudes)) if len(amplitudes) > 0 else 0.0,
    }


def get_amplitude_changes(model: LieSolver) -> np.ndarray:
    """Compute amplitude changes between consecutive steps.
    
    Args:
        model (LieSolver): Model instance.
    
    Returns:
        np.ndarray: Pairwise differences (shape varies based on term changes).
                   For each step, returns the L2 norm of amplitude changes.
    """
    if len(model.history['amplitudes']) < 2:
        return np.array([])
    
    changes = []
    for i in range(1, len(model.history['amplitudes'])):
        prev_a = model.history['amplitudes'][i-1]
        curr_a = model.history['amplitudes'][i]
        
        # For arrays of different lengths, only compare common indices
        min_len = min(len(prev_a), len(curr_a))
        if min_len > 0:
            change = np.linalg.norm(curr_a[:min_len] - prev_a[:min_len])
        else:
            change = 0.0
        changes.append(change)
    
    return np.array(changes)


def plot_amplitude_evolution(model: LieSolver, figsize: Tuple[float, float] = (12, 6), 
                             margin_fraction: float = 0.5) -> None:
    """Plot amplitude evolution across fitting history with relative amplitudes.
    
    Visualizes how each term's amplitude (normalized by total) evolves throughout
    the fitting process. Each term is shown at a fixed x-position, with points
    scattered horizontally for visibility. Transparency indicates age (older = more
    transparent), color indicates final state (red = final, black = history).
    
    Args:
        model (LieSolver): Model instance with history tracking enabled.
        figsize (Tuple[float, float]): Figure size as (width, height). Default (12, 6).
        margin_fraction (float): Margin as fraction of y-range from percentiles.
                                Default 0.1 (10% on each side).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    history_amplitudes = model.history['amplitudes']
    history_steps = model.history['step_type']
    
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
    plt.show()
