from liesolver.loader import parse
from liesolver.utils.logging import log_output
from liesolver.utils.io import configure_logging, create_output_dir
from liesolver.trainer import Trainer

def main():
    cfg = parse()
    out_dir = create_output_dir(cfg["experiment_name"],
                                    suffix=cfg.get("suffix", ""))
    configure_logging(out_dir / 'run.log')
    with log_output():
        trainer = Trainer(cfg, out_dir)
        trainer.init_model()
        trainer.fit()

if __name__ == '__main__':
    main()

