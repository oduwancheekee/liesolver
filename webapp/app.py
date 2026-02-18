"""Simple Flask webapp for LieSolver."""
import sys
import os
import re

# Ensure LieSolver is importable and CWD is project root
LIESOLVER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LIESOLVER_ROOT)
os.chdir(LIESOLVER_ROOT)

import json
import threading
import queue
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_file

import yaml
from liesolver.utils import create_output_dir, configure_logging
from liesolver.trainer import Trainer

app = Flask(__name__)

# --------------- Global state ---------------
stop_event = threading.Event()
current_output_dir = None
fit_thread = None

# Broadcast: each SSE client gets its own queue
_subscribers = []
_subscribers_lock = threading.Lock()

ALLOWED_PDE_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def broadcast(msg: dict):
    """Send a message to every active SSE subscriber."""
    with _subscribers_lock:
        for q in _subscribers:
            q.put(msg)


def get_config_files():
    configs_dir = Path(__file__).parent.parent / "configs"
    return sorted([f.name for f in configs_dir.glob("*.yaml")])


def validate_config(cfg: dict) -> str | None:
    """Return error string if config is unsafe, else None."""
    if not isinstance(cfg, dict):
        return "Config must be a YAML mapping"
    pde = cfg.get("pde", "")
    if not ALLOWED_PDE_RE.match(str(pde)):
        return f"Invalid PDE name: '{pde}'. Must be alphanumeric/underscore."
    pdes_dir = Path(LIESOLVER_ROOT) / "liesolver" / "pdes"
    if not (pdes_dir / f"{pde}.py").exists():
        available = [f.stem for f in pdes_dir.glob("*.py") if f.stem != "__init__"]
        return f"Unknown PDE '{pde}'. Available: {available}"
    return None


# --------------- Routes ---------------

@app.route("/")
def index():
    return render_template("index.html", configs=get_config_files())


@app.route("/api/config/<filename>")
def get_config(filename):
    config_path = Path(__file__).parent.parent / "configs" / filename
    if not config_path.exists():
        return "Config not found", 404
    return Response(config_path.read_text(), mimetype="text/plain")


@app.route("/api/fit", methods=["POST"])
def start_fit():
    global current_output_dir, fit_thread

    if fit_thread and fit_thread.is_alive():
        return jsonify({"error": "Fit already running"}), 400

    try:
        cfg = yaml.safe_load(request.json.get("config_yaml", ""))
    except yaml.YAMLError as e:
        return jsonify({"error": f"Invalid YAML: {e}"}), 400

    err = validate_config(cfg)
    if err:
        return jsonify({"error": err}), 400

    stop_event.clear()
    fit_thread = threading.Thread(target=run_fit, args=(cfg,), daemon=True)
    fit_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def stop_fit():
    stop_event.set()
    return jsonify({"status": "stopping"})


@app.route("/api/progress")
def progress_stream():
    """SSE endpoint. Each connection gets its own queue via broadcast."""
    q = queue.Queue()
    with _subscribers_lock:
        _subscribers.append(q)

    def generate():
        try:
            while True:
                try:
                    msg = q.get(timeout=20)
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    continue
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
        finally:
            with _subscribers_lock:
                _subscribers.remove(q)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/figures")
def get_figures():
    if current_output_dir is None:
        return jsonify({"figures": []})
    figures = sorted(f.name for f in Path(current_output_dir).glob("*.png"))
    return jsonify({"figures": figures, "output_dir": str(current_output_dir)})


@app.route("/api/figure/<filename>")
def get_figure(filename):
    if current_output_dir is None:
        return "No output dir", 404
    fig_path = Path(current_output_dir) / filename
    if not fig_path.exists():
        return "Figure not found", 404
    return send_file(fig_path, mimetype="image/png")


# --------------- Fit logic ---------------

def run_fit(cfg: dict):
    global current_output_dir
    try:
        out_dir = create_output_dir(cfg["experiment_name"], suffix=cfg.get("suffix", ""))
        current_output_dir = Path(LIESOLVER_ROOT) / out_dir
        configure_logging(out_dir / "run.log")

        broadcast({"type": "info", "msg": f"Output: {current_output_dir}"})

        trainer = Trainer(cfg, out_dir)
        trainer.init_model()
        trainer._stop_event = stop_event
        _run_fit_loop(trainer)

        broadcast({"type": "done", "msg": "Fitting complete", "output_dir": str(current_output_dir)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        broadcast({"type": "error", "msg": str(e)})


def _run_fit_loop(trainer: Trainer):
    import numpy as np

    fit_cfg = trainer.fit_cfg
    max_bricks = fit_cfg.get("max_bricks", 20)
    mse_tol = float(fit_cfg.get("mse_tol", 1e-3))
    nfev_global = fit_cfg.get("nfev_global", 2)
    nfev_batch = fit_cfg.get("nfev_batch", 10)
    global_every = fit_cfg.get("global_every", 10)
    batch_size = fit_cfg.get("batch_size", 5)
    pool_size = fit_cfg.get("pool_size", 100)

    model, state, data = trainer.model, trainer.state, trainer.data

    broadcast({"type": "header", "msg": "Action          MSEtrain   MSEtest   MSEdomain Trafos•start_fun"})

    for i in range(max_bricks):
        if stop_event.is_set():
            broadcast({"type": "info", "msg": f"[Stopped] Early stop at {len(model.bricks)} bricks"})
            break

        model.add_best_brick(pool_size=pool_size)
        state.log(model, data, step_type="add_best_brick")

        brick = model.bricks[-1]
        a = model.amplitudes[-1]
        sign = "+" if a > 0 else "-"
        amp = f"{sign}{abs(a):.2f}" if abs(a) >= 0.01 else f"{sign}{abs(a):.0e}"

        broadcast({
            "type": "add",
            "msg": f"Add {i+1} a:{amp} | {state.train_mse_hist[-1]:.2e} | {state.test_mse_hist[-1]:.1e} | {state.domain_mse_hist[-1]:.1e} | {brick.family}",
        })

        K = len(model.bricks)
        if ((i + 1) % batch_size) == 0:
            active_idx = list(range(max(0, K - batch_size), K))
            model.refine(max_nfev=nfev_batch, active_idx=active_idx)
            state.log(model, data, step_type="refine_batch")
            broadcast({"type": "refine", "msg": f"Refine batch | {model.mse:.2e} | {state.test_mse_hist[-1]:.1e} | {state.domain_mse_hist[-1]:.1e}"})

        if (((i + 1) % global_every) == 0) or ((i + 1) == max_bricks):
            model.refine(max_nfev=nfev_global)
            state.log(model, data, step_type="refine_all")
            broadcast({"type": "refine", "msg": f"Refine all {K} | {model.mse:.2e} | {state.test_mse_hist[-1]:.1e} | {state.domain_mse_hist[-1]:.1e}"})

        if model.mse <= mse_tol:
            broadcast({"type": "info", "msg": f"Reached MSE tolerance {mse_tol}"})
            break

    # Save outputs
    model.save(trainer.out_dir)
    state.save(trainer.out_dir / f"{trainer.experiment_name}-fit_state.npz")

    from liesolver.plotting import plot_2d_domain, plot_ic_bc, plot_fit_history
    plot_fit_history(state, save_to=trainer.out_dir / f"{trainer.experiment_name}-fit_history.png", compact=False)
    plot_ic_bc(model, data, filepath=trainer.out_dir / f"{trainer.experiment_name}-ic_bc_plot.png", compact=True)
    plot_ic_bc(model, data, decompose=True, filepath=trainer.out_dir / f"{trainer.experiment_name}-ic_bc_plot-decompose.png", compact=True)
    plot_2d_domain(model, data, filepath=trainer.out_dir / f"{trainer.experiment_name}-domain_plot.png", compact=True)


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
