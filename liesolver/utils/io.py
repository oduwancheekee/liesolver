from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import yaml

import logging
import sys

def configure_logging(log_path: str = "run.log") -> None:
    """Configure application logging to output to stdout and a log file.
    
    Args:
        log_path: Path to the log file to append to.
    """
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
    """Create a run-specific output directory: root/DATE/TIME_experiment[_suffix].
    
    Args:
        experiment_name: Name used in the directory prefix.
        root: Base output directory (created if missing).
        suffix: Optional extra tag appended after the experiment name.
    
    Returns:
        Path: Path to the created directory.
    """
    date_part, time_part = _now()
    parts = [time_part, experiment_name]
    if suffix:
        parts.append(suffix)
    out_dir = Path(root) / date_part / "_".join(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _now() -> tuple[str, str]:
    """Get current datetime as (YYYY-MM-DD, HH-MM-SS)."""
    ts = datetime.now()
    return ts.date().isoformat(), ts.strftime("%H-%M-%S")


def save_config(cfg: Dict, out_dir: Path, name: str = "config.yaml") -> Path:
    """Save a configuration dictionary as YAML into the output directory.
    
    Args:
        cfg: Configuration mapping to serialize.
        out_dir: Destination folder where the file is written.
        name: File name to use (default: 'config.yaml').
    
    Returns:
        Path: Path to the written YAML file.
    """
    path = out_dir / name
    with open(path, "w") as fp:
        yaml.safe_dump(cfg, fp)
    return path