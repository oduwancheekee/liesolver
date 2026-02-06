# loader.py
import argparse
import ast
import sys
from typing import List, Optional, Sequence, Union

import yaml


def parse_value(text: str):
    """Parse a string value into a Python type safely.
    
    Args:
        text: Raw value string (e.g., '1', '0.1', 'true', '[1,2]').
    
    Returns:
        Parsed Python value or original string if parsing fails.
    """
    s = text.strip()
    low = s.lower()
    if low in ("true", "false", "none", "null"):
        return {"true": True, "false": False, "none": None, "null": None}[low]
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def parse_path(path: str) -> List[Union[str, int]]:
    """Parse a dotted/bracketed path into tokens.
    
    Args:
        path: Key path like 'data.range_dim[1][1]' or 'data.num_ic'.
    
    Returns:
        List[Union[str, int]]: Tokens (str for dict keys, int for list indices).
    """
    tokens: List[Union[str, int]] = []
    buf = ""
    i = 0
    n = len(path)
    while i < n:
        c = path[i]
        if c == '.':
            if buf:
                tokens.append(buf)
                buf = ""
            i += 1
        elif c == '[':
            if buf:
                tokens.append(buf)
                buf = ""
            j = path.find(']', i + 1)
            if j == -1:
                raise ValueError(f"Missing closing bracket in path: {path}")
            idx_str = path[i + 1 : j].strip()
            if not idx_str.isdigit():
                raise ValueError(f"List index must be a non-negative integer in: {path}")
            tokens.append(int(idx_str))
            i = j + 1
        else:
            buf += c
            i += 1
    if buf:
        tokens.append(buf)
    if not tokens:
        raise ValueError(f"Invalid path: {path}")
    return tokens


def set_by_path(config: dict, tokens: Sequence[Union[str, int]], value) -> None:
    """Set a value in a nested config following path tokens.
    
    Args:
        config: Root configuration dictionary.
        tokens: Path tokens (str keys, int indices).
        value: Value to assign.
    """
    cur = config
    # Traverse to parent of the target
    for tok in tokens[:-1]:
        if isinstance(tok, int):
            try:
                cur = cur[tok]
            except (TypeError, IndexError):
                raise KeyError(f"Invalid list index in path: {tokens}")
        else:
            try:
                cur = cur[tok]
            except (TypeError, KeyError):
                raise KeyError(f"Unknown key in path: {tokens}")

    last = tokens[-1]
    if isinstance(last, int):
        try:
            cur[last] = value
        except (TypeError, IndexError):
            raise KeyError(f"Invalid list index at assignment in path: {tokens}")
    else:
        if not isinstance(cur, dict):
            raise KeyError(f"Expected dict at assignment in path: {tokens}")
        if last not in cur:
            raise KeyError(f"Unknown key at assignment in path: {tokens}")
        cur[last] = value


def apply_overrides(config: dict, overrides: Sequence[str]) -> None:
    """Apply CLI-style overrides (key=val) to a config.
    
    Args:
        config: Base config dict to modify in place.
        overrides: Overrides like 'data.num_ic=2000'.
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be 'key=val': {ov}")
        key, val = ov.split("=", 1)
        tokens = parse_path(key.strip())
        set_by_path(config, tokens, parse_value(val))


def load_config(config_path: str, overrides: Optional[Sequence[str]] = None) -> dict:
    """Load YAML config and apply optional overrides.
    
    Args:
        config_path: Path to YAML file.
        overrides: Optional list of 'path=value' strings.
    
    Returns:
        dict: Final configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if overrides:
        apply_overrides(config, overrides)
    return config


def parse_cli(default: str = "configs/example.yaml") -> dict:
    """Parse CLI, load config, and apply overrides.
    
    Args:
        default: Default config path if none is provided.
    
    Returns:
        dict: Final configuration dictionary.
    """
    parser = argparse.ArgumentParser(description="Fits LieSolver with config-set parameters")
    parser.add_argument("config", nargs="?", default=default, help="Path to config file")
    args, unknown = parser.parse_known_args()
    overrides = [u for u in unknown if "=" in u]
    return load_config(args.config, overrides=overrides)


def get_cli_overrides(argv: Optional[Sequence[str]] = None) -> List[str]:
    """Extract 'key=value' overrides from argv.
    
    Args:
        argv: Argument list; defaults to sys.argv[1:].
    
    Returns:
        List[str]: Overrides like ['seed=1', 'data.range_dim[1][1]=0.2'].
    """
    args = list(sys.argv[1:] if argv is None else argv)
    return [a for a in args if "=" in a]


def dump_overrides_yaml(file_path: str, overrides: Sequence[str]) -> None:
    """Dump overrides to a YAML file as path: parsed_value pairs.
    
    Args:
        file_path: Destination YAML path.
        overrides: Overrides like ['seed=1', 'data.num_ic=2000'].
    """
    mapping = {}
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be 'key=val': {ov}")
        key, val = ov.split("=", 1)
        mapping[key.strip()] = parse_value(val)
    with open(file_path, "w") as f:
        yaml.safe_dump(mapping, f, sort_keys=False)
