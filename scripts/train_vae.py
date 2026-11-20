"""Train one VAE of a model pair.

    python scripts/train_vae.py --config-name vae_fmnist_gauss training.seed=42 general.run_name=vae_a
    python scripts/train_vae.py --config-name vae_fmnist_gauss training.seed=123 general.run_name=vae_b

The checkpoint goes to configs' training.checkpoint (checkpoints/<dataset>_<prior>/<run_name>):
best_model.pickle, last_checkpoint.pickle, config.yml and metrics.csv.
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import os
import pathlib as pl

import hydra
import jax
from flax import nnx
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from ngma.data import get_arrays
from ngma.training.vae_trainer import MetricsLogger
from ngma.utils import set_seed, load_obj


def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.training.checkpoint, "run.log")),
            logging.StreamHandler(),
        ],
    )
    logging.info(OmegaConf.to_yaml(cfg))

    set_seed(cfg["training"]["seed"])
    rng = jax.random.PRNGKey(cfg["training"]["seed"])

    (
        train_images, train_labels,
        val_images, val_labels,
        test_images, test_labels,
    ) = get_arrays(
        cfg["datamodule"]["dataset_name"],
        cfg["datamodule"]["dataset_root"],
        val_fraction=cfg["datamodule"].get("val_fraction", 0.1),
        seed=cfg["training"]["seed"],
    )

    train_images = jax.device_put(train_images)
    val_images = jax.device_put(val_images)
    test_images = jax.device_put(test_images)

    model = load_obj(cfg["model"]["class_name"])(
        opts=cfg["model"]["params"], rngs=nnx.Rngs(rng)
    )

    if getattr(model.opts, "prior", "standard") == "vamp":
        model.init_pseudo_inputs(train_images)
        logging.info(
            f"VampPrior: initialised {model.opts.vamp_K} pseudo-inputs from training samples"
        )

    logger = MetricsLogger(os.path.join(cfg.training.checkpoint, "metrics.csv"))
    trainer = load_obj(cfg["training"]["class_name"])(model, cfg, logger)
    trainer.train_model(train_images, val_images, cfg["training"]["num_epochs"])
    # Report on the checkpoint the later stages load, not on the last epoch.
    trainer.restore_best()
    test_stats = trainer.eval_model(test_images, cfg["training"]["num_epochs"] + 1, prefix="test")
    logging.info(f"Test stats (selected checkpoint): {test_stats}")


@hydra.main(config_path="../configs", config_name="vae_mnist_gauss", version_base=None)
def launch(cfg: DictConfig):
    cfg.training.checkpoint = os.path.join(get_original_cwd(), cfg.training.checkpoint)
    cfg.datamodule.dataset_root = os.path.join(get_original_cwd(), cfg.datamodule.dataset_root)
    ckpt = pl.Path(cfg.training.checkpoint)
    if (ckpt / "best_model.pickle").exists():
        raise SystemExit(
            f"REFUSING: {ckpt}/best_model.pickle exists. Point training.checkpoint at a "
            f"fresh directory or move the old run aside."
        )
    ckpt.mkdir(parents=True, exist_ok=True)
    main(cfg)


if __name__ == "__main__":
    launch()
