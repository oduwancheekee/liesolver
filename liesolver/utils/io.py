from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import torch
import yaml

import logging
import sys

def configure_logging(log_path: str = "run.log") -> None:
    # fmt = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    fmt = "[%(asctime)s] - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),   # console
            logging.FileHandler(log_path, "a"),  # file
        ],
    )

def create_output_dir(
    experiment_name: str,
    *,
    root: Union[str, Path] = "outputs",
    suffix: str = "",
) -> Path:
    """
    Make run-specific directory   outputs/DATE/TIME_experiment_suffix

    Parameters
    ----------
    experiment_name : str
    root            : base folder
    suffix          : optional extra tag (e.g. hyper-run id)

    Returns
    -------
    Path to the freshly created directory (parents=True, exist_ok=True).
    """
    date_part, time_part = _now()
    parts = [time_part, experiment_name]
    if suffix:
        parts.append(suffix)
    out_dir = Path(root) / date_part / "_".join(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _now() -> tuple[str, str]:
    """Return (YYYY-MM-DD, HH-MM-SS)."""
    ts = datetime.now()
    return ts.date().isoformat(), ts.strftime("%H-%M-%S")


def save_config(cfg: Dict, out_dir: Path, name: str = "config.yaml") -> Path:
    path = out_dir / name
    with open(path, "w") as fp:
        yaml.safe_dump(cfg, fp)
    return path


def save_checkpoint(model: Any, out_dir: Path, tag: str = "last") -> Path:
    """
    Save *state_dict* if the object has it, otherwise save whole object.

    Example filenames: model_last.pt, model_epoch2000.pt
    """
    path = out_dir / f"model_{tag}.pt"
    obj = model.state_dict() if hasattr(model, "state_dict") else model
    torch.save(obj, path)
    return path


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """
    Store a matplotlib Figure in *out_dir/name* (dpi=300, tight layout) and close it.
    """
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path