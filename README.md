# LieSolver

A solver for initial-boundary value problems (IBVPs) of linear homogeneous PDEs using Lie symmetries. The PDE is satisfied exactly by construction, optimization only fits initial and boundary conditions (IBCs).

## Requirements

- Python ≥ 3.12

## Installation

Unzip the archive and install:

```bash
cd LieSolver
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quick Start

```bash
python main.py configs/heat_1d.yaml
```

### Output

Results are saved to `outputs/<date>/<time>_<experiment_name>/`:

| File | Description |
|------|-------------|
| `config.yaml` | Full experiment configuration |
| `model_*.npz` | Trained model (brick families, parameters, amplitudes) |
| `*-fit_state.npz` | Training history (MSE, amplitudes per step) |
| `*-fit_history.png` | MSE convergence vs number of bricks |
| `*-ic_bc_plot.png` | IC/BC fit comparison (model vs target) |
| `*-ic_bc_plot-decompose.png` | Individual brick contributions on IC/BC |
| `*-domain_plot.png` | 2D solution field with error |
| `run.log` | Console output log |


## CLI Overrides

Override any config value from command line:

```bash
python main.py configs/wave_1d.yaml seed=42 fit.max_bricks=30 fit.mse_tol=1e-5
```

## Project Structure

```
liesolver/
├── model.py        # LieSolver, BrickFamily, Transformation
├── trainer.py      # Training loop
├── dataloader.py   # Constraint sampling and data generation
├── constraints.py  # IC/BC constraint data structures
├── pdes/           # PDE modules (heat_1d, wave_1d, ...)
└── utils/          # Config parsing, logging, I/O
configs/            # Experiment YAML files
```

## Demo

See `demo.ipynb` for an interactive example that trains a model and visualizes results.

## License

License information withheld for anonymous peer review. Use this code for review purposes only. Do not share.
