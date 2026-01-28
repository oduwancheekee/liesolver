"""Constraint data structures for IC/BC fitting with derivative support."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class Constraint:
    """A single constraint: targets for function value or derivative at sample points.
    
    Attributes:
        x: Sample coordinates, shape (N, dim).
        y: Target values, shape (N,).
        deriv_order: Derivative order per coordinate dimension.
            (0, 0) = function value u(x,t)
            (1, 0) = ∂u/∂x
            (0, 1) = ∂u/∂t
            (2, 0) = ∂²u/∂x²
            etc.
        weight: Loss weight for this constraint.
        name: Human-readable name for logging.
    """
    x: np.ndarray
    y: np.ndarray
    deriv_order: Tuple[int, ...]
    weight: float = 1.0
    name: str = ""
    
    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64).ravel()
        self.deriv_order = tuple(self.deriv_order)
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"x and y must have same number of samples: {self.x.shape[0]} vs {self.y.shape[0]}")
    
    @property
    def n_samples(self) -> int:
        return self.x.shape[0]
    
    @property
    def is_value(self) -> bool:
        """True if this is a function value constraint (no derivatives)."""
        return all(d == 0 for d in self.deriv_order)
    
    def __repr__(self) -> str:
        return f"Constraint({self.name!r}, n={self.n_samples}, deriv={self.deriv_order}, w={self.weight})"


@dataclass 
class ConstraintSet:
    """Collection of constraints for fitting.
    
    Provides unified interface to stack constraints for linear algebra operations.
    """
    constraints: List[Constraint] = field(default_factory=list)
    
    def add(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)
    
    @property
    def n_constraints(self) -> int:
        return len(self.constraints)
    
    @property
    def total_samples(self) -> int:
        return sum(c.n_samples for c in self.constraints)
    
    def get_weighted_targets(self) -> np.ndarray:
        """Stack all weighted targets into a single vector."""
        if not self.constraints:
            return np.array([])
        parts = [np.sqrt(c.weight) * c.y for c in self.constraints]
        return np.concatenate(parts)
    
    def summary(self) -> str:
        lines = [f"ConstraintSet with {self.n_constraints} constraints, {self.total_samples} total samples:"]
        for c in self.constraints:
            lines.append(f"  {c}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"ConstraintSet({self.n_constraints} constraints, {self.total_samples} samples)"
    
    def __iter__(self):
        return iter(self.constraints)
    
    def __len__(self):
        return len(self.constraints)
