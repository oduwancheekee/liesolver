"""Metrics and analysis functions for LieSolver models."""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Tuple
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .model import LieSolver
    from .dataloader import DataLoader


class Metric(ABC):
    """Base class for metrics that can be tracked during training.
    
    Users can inherit from this class to define custom metrics that are
    automatically tracked during the training process.
    
    Example:
        class ConditionNumberMetric(Metric):
            def __init__(self):
                super().__init__(name='condition_number')
            
            def compute(self, model: LieSolver, data: DataLoader) -> float:
                F = model.feature_matrix(model.bricks)
                return condition_number(model, F)
    """
    
    def __init__(self, name: str):
        """Initialize metric with a name.
        
        Args:
            name (str): Name of the metric (used for storage and plotting).
        """
        self.name = name
    
    @abstractmethod
    def compute(self, model: LieSolver, data: DataLoader) -> float:
        """Compute metric value given model and data.
        
        Args:
            model (LieSolver): The model instance.
            data (DataLoader): The data loader instance.
        
        Returns:
            float: The computed metric value.
        """
        raise NotImplementedError("Subclasses must implement compute()")


class ConditionNumberMetric(Metric):
    """Built-in metric for tracking condition number of feature matrix."""
    
    def __init__(self):
        super().__init__(name='condition_number')
    
    def compute(self, model: LieSolver, data: DataLoader) -> float:
        """Compute condition number of feature matrix F."""
        F = model.feature_matrix(model.bricks)
        return condition_number(model, F)


class ConditionNumberMetric(Metric):
    """Metric that computes the condition number of the feature matrix."""
    
    def __init__(self):
        super().__init__(name='condition_number')
    
    def compute(self, model: LieSolver, data: DataLoader) -> float:
        """Compute condition number of feature matrix."""
        return condition_number(model)


class PredictionAlignmentMetric(Metric):
    """Metric that measures misalignment (1 - cosine similarity) between prediction and target.
    
    Computes 1 - cos(x) where cos(x) = <F·a, Y> / (||F·a|| · ||Y||), 
    the angle between the prediction vector F·a and target vector Y.
    
    Values:
        - 0.0: Perfect alignment (vectors point in same direction)
        - 1.0: Orthogonal (no correlation)
        - 2.0: Opposite directions
    
    This metric is complementary to MSE:
        - MSE measures magnitude of error
        - Misalignment measures direction/alignment
    
    Note: Returns 1 - cosine_similarity so that perfect alignment is near 0,
    allowing log scale visualization of small differences during training.
    """
    
    def __init__(self):
        super().__init__(name='prediction_alignment')
    
    def compute(self, model: LieSolver, data: DataLoader) -> float:
        """Compute misalignment (1 - cosine similarity) between prediction and target."""
        pred = model(data.train_x)
        target = data.train_y
        
        dot_product = np.dot(pred, target)
        norm_pred = np.linalg.norm(pred)
        norm_target = np.linalg.norm(target)
        
        # Avoid division by zero
        if norm_pred < 1e-15 or norm_target < 1e-15:
            return 1.0  # Maximum misalignment if either vector is zero
        
        cosine_sim = dot_product / (norm_pred * norm_target)
        return float(1.0 - cosine_sim)
    

class PredictionAlignmentMetric2(Metric):
    """Metric that measures misalignment (1 - cosine similarity) between prediction and target.
    
    Computes 1 - cos(x) where cos(x) = <F·a, Y> / (||F·a|| · ||Y||), 
    the angle between the prediction vector F·a and target vector Y.
    
    Values:
        - 0.0: Perfect alignment (vectors point in same direction)
        - 1.0: Orthogonal (no correlation)
        - 2.0: Opposite directions
    
    This metric is complementary to MSE:
        - MSE measures magnitude of error
        - Misalignment measures direction/alignment
    
    Note: Returns 1 - cosine_similarity so that perfect alignment is near 0,
    allowing log scale visualization of small differences during training.
    """
    
    def __init__(self):
        super().__init__(name='prediction_alignment2')
    
    def compute(self, model: LieSolver, data: DataLoader) -> float:
        """Compute misalignment (1 - cosine similarity) between prediction and target."""
        pred = model(data.train_x)
        target = data.train_y
        
        dot_product = np.dot(pred, target)
        norm_pred = np.linalg.norm(pred)
        norm_target = np.linalg.norm(target)
        
        # Avoid division by zero
        if norm_pred < 1e-15 or norm_target < 1e-15:
            return 1.0  # Maximum misalignment if either vector is zero
        
        cosine_sim = dot_product / (norm_pred * norm_target)
        return float(norm_pred)



def condition_number(model: LieSolver) -> float:
    """
    Compute the condition number of F^T F (ratio of largest to smallest eigenvalue).
    
    A low condition number (close to 1) indicates a well-conditioned system.
    A high condition number indicates numerical instability and potential ill-conditioning.
    
    Args:
        model (LieSolver): Model instance.
    
    Returns:
        float: Condition number κ(F^T F) = λ_max / λ_min.
    """
    F = model.feature_matrix(model.bricks)
    
    if F.shape[1] == 0:
        return np.inf
    
    FtF = F.T @ F
    eigvals = np.linalg.eigvalsh(FtF)
    eigvals = np.abs(eigvals[eigvals > 1e-15])  # Filter near-zero eigenvalues
    
    if len(eigvals) == 0:
        return np.inf
    
    return float(np.max(eigvals) / np.min(eigvals))


def matrix_rank(model: LieSolver, tol: float = 1e-10) -> int:
    """
    Compute the numerical rank of the feature matrix F.
    
    A full rank matrix has rank equal to min(L, M).
    Rank deficiency indicates linear dependence among brick functions.
    
    Args:
        model (LieSolver): Model instance.
        tol (float): Threshold for considering singular values as non-zero.
    
    Returns:
        int: Numerical rank of F.
    """
    F = model.feature_matrix(model.bricks)
    
    if F.shape[1] == 0:
        return 0
    
    _, s, _ = np.linalg.svd(F, full_matrices=False)
    return int(np.sum(s > tol))


def eigenvalue_distribution(model: LieSolver) -> Tuple[np.ndarray, dict]:
    """
    Compute eigenvalues of F^T F and statistics on their distribution.
    
    For a well-conditioned problem with orthogonal bricks, eigenvalues should be
    clustered and relatively uniform. High variance in eigenvalues indicates
    some bricks are much more important than others.
    
    Args:
        model (LieSolver): Model instance.
    
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
    F = model.feature_matrix(model.bricks)
    
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


def brick_orthogonality(model: LieSolver) -> Tuple[float, dict]:
    """
    Check orthogonality of brick functions using Hilbert space inner product.
    
    For a domain [a, b], the Hilbert space inner product is:
        <f_i, f_j> ≈ Σ f_i(x_k) * f_j(x_k) * dx  (quadrature approximation)
    
    The normalized metric is:
        max_{i ≠ j} |<f_i, f_j>| / (||f_i|| * ||f_j||)
    
    A value close to 0 indicates near-orthogonal bricks (good).
    A value close to 1 indicates significant overlap (problematic).
    
    Args:
        model (LieSolver): Model instance.
    
    Returns:
        Tuple[float, dict]:
            - max_overlap: maximum normalized inner product between distinct bricks
            - stats dict with keys:
                - 'mean_overlap': mean of all pairwise normalized overlaps
                - 'min_overlap': minimum normalized overlap
                - 'max_overlap': maximum normalized overlap
                - 'gram_matrix': full Gram matrix (M x M) of normalized overlaps
    """
    X = model.X
    
    if len(model.bricks) == 0:
        return 0.0, {'mean_overlap': 0.0, 'min_overlap': 0.0, 'max_overlap': 0.0, 'gram_matrix': np.array([])}
    
    M = len(model.bricks)
    
    # Compute brick function evaluations
    brick_evals = []
    for brick in model.bricks:
        phi = brick.family.eval(X, brick.params)
        brick_evals.append(phi)
    brick_evals = np.array(brick_evals)  # (M, L)
    
    # Compute norms (L2 norm over data points)
    norms = np.linalg.norm(brick_evals, axis=1)  # (M,)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    
    # Compute Gram matrix (unnormalized inner products)
    gram_raw = brick_evals @ brick_evals.T  # (M, M)
    
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


def get_history_summary(fit_state) -> dict:
    """Get summary statistics of the fitting history.
    
    Args:
        fit_state (FitState): FitState instance from trainer.
    
    Returns:
        dict: Summary with keys:
            - 'n_steps': total number of steps recorded
            - 'step_types': list of step types
            - 'mse_initial': MSE at start
            - 'mse_final': MSE at end
            - 'mse_improvement': percentage improvement
            - 'n_bricks_added': number of bricks added
            - 'n_refines': number of refinement steps
            - 'n_removals': number of bricks removed
    """
    if not fit_state.train_mse_hist:
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
    
    mse_arr = np.array(fit_state.train_mse_hist)
    step_types = fit_state.step_type_hist
    
    mse_initial = mse_arr[0] if len(mse_arr) > 0 else np.nan
    mse_final = mse_arr[-1] if len(mse_arr) > 0 else np.nan
    mse_improvement = 100.0 * (mse_initial - mse_final) / (mse_initial + 1e-12) if mse_initial > 0 else 0.0
    
    return {
        'n_steps': len(fit_state.train_mse_hist),
        'step_types': step_types,
        'mse_initial': float(mse_initial),
        'mse_final': float(mse_final),
        'mse_improvement': float(mse_improvement),
        'n_terms_added': sum(1 for st in step_types if 'add' in st),
        'n_refines': sum(1 for st in step_types if 'refine' in st),
        'n_removals': sum(1 for st in step_types if 'remove' in st),
    }


def get_step_info(fit_state, step_idx: int) -> dict:
    """Get detailed information about a specific step.
    
    Args:
        fit_state (FitState): FitState instance from trainer.
        step_idx (int): Index of the step in history.
    
    Returns:
        dict: Information with keys:
            - 'step_type': type of step
            - 'mse': MSE at this step
            - 'n_bricks': number of bricks at this step
            - 'amplitudes': amplitude vector
            - 'brick_params': parameters of all bricks
            - 'amplitude_norms': L2 norms of amplitudes
    """
    if step_idx < 0 or step_idx >= len(fit_state.train_mse_hist):
        raise IndexError(f"step_idx {step_idx} out of range [0, {len(fit_state.train_mse_hist)-1}]")
    
    amplitudes = fit_state.amplitudes_hist[step_idx]
    term_params = fit_state.term_params_hist[step_idx]
    
    return {
        'step_type': fit_state.step_type_hist[step_idx],
        'mse': float(fit_state.train_mse_hist[step_idx]),
        'n_terms': int(fit_state.nterms_hist[step_idx]),
        'amplitudes': amplitudes,
        'term_params': term_params,
        'amplitude_norms': float(np.linalg.norm(amplitudes)) if len(amplitudes) > 0 else 0.0,
    }


def get_amplitude_changes(fit_state) -> np.ndarray:
    """Compute amplitude changes between consecutive steps.
    
    Args:
        fit_state (FitState): FitState instance from trainer.
    
    Returns:
        np.ndarray: Pairwise differences (shape varies based on term changes).
                   For each step, returns the L2 norm of amplitude changes.
    """
    if len(fit_state.amplitudes_hist) < 2:
        return np.array([])
    
    changes = []
    for i in range(1, len(fit_state.amplitudes_hist)):
        prev_a = fit_state.amplitudes_hist[i-1]
        curr_a = fit_state.amplitudes_hist[i]
        
        # For arrays of different lengths, only compare common indices
        min_len = min(len(prev_a), len(curr_a))
        if min_len > 0:
            change = np.linalg.norm(curr_a[:min_len] - prev_a[:min_len])
        else:
            change = 0.0
        changes.append(change)
    
    return np.array(changes)



