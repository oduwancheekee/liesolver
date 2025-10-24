import yaml
import os
import argparse

def parse_overrides(overrides):
    """
    Parse key=value style overrides into a dictionary.
    Supports nested keys like `training.0.lr=0.001`.
    """
    config_updates = {}
    for override in overrides:
        key, value = override.split("=", 1)
        keys = key.split(".")  # Support nested keys
        sub_config = config_updates
        for k in keys[:-1]:
            sub_config = sub_config.setdefault(k, {})
        # Attempt to infer type of the value (int, float, bool, etc.)
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            pass  # If eval fails, treat as a string
        sub_config[keys[-1]] = value
    return config_updates


def update_nested_dict(base, updates):
    """
    Recursively update a nested dictionary `base` with values from `updates`.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            update_nested_dict(base[key], value)
        else:
            base[key] = value


def load_config(config_path, overrides=None):
    """
    Load and parse a YAML configuration file, applying any overrides.

    Args:
        config_path (str): Path to the YAML config file.
        overrides (list): List of key=value overrides (optional).

    Returns:
        dict: Final configuration dictionary.
    """
    # Load YAML config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Parse and apply overrides if provided
    if overrides:
        config_updates = parse_overrides(overrides)
        update_nested_dict(config, config_updates)

    # Perform basic validation or add defaults (optional)
    validate_and_add_defaults(config)

    return config


def validate_and_add_defaults(config: dict):
    """
    Perform basic validation on the config and add defaults where necessary.

    Args:
        config (dict): The parsed configuration dictionary.
    """
    # Ensure output directory exists
    output_dir = config.get("output_dir", "outputs/")
    os.makedirs(output_dir, exist_ok=True)

    # Validate device
    if config.get("device", None) not in ["cpu", "cuda", "mps", None]:
        raise ValueError("Invalid device specified. Must be 'cpu', 'cuda' or 'mps'.")

    # Validate layers and prune
    layers = config.get("layers", 0)
    prune = config.get("prune", {})
    if not isinstance(prune, dict):
        raise ValueError("`prune` must be a dictionary mapping layer indices to node indices.")
    for layer, nodes in prune.items():
        if not (0 <= int(layer) < layers):
            raise ValueError(f"Invalid layer index in prune: {layer}. Must be in range [0, {layers-1}].")
        if not all(isinstance(node, int) for node in nodes):
            raise ValueError(f"Pruned nodes must be integers in layer {layer}: {nodes}.")

    # Add any other validation rules or defaults as needed
    # Example: Set a default random seed if not specified
    if "seed" not in config:
        config["seed"] = 0


def parse(default="configs/example.yaml") -> dict:
    """Parse CLI args and return the loaded YAML/JSON config as dict."""
    parser = argparse.ArgumentParser(description="Train Lie-Symmetry model")
    parser.add_argument(
        "config",
        nargs="?",
        default=default,
        help="Path to config file",
    )
    return load_config(parser.parse_args().config)
