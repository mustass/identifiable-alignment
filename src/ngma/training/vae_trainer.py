"""Training loop of the VAEs.

One epoch is one jitted `nnx.scan` over the shuffled batches. Every `eval_every` epochs the
validation ELBO at the final beta decides the checkpoint that is kept (`best_model.pickle`);
during the beta warm-up the latest checkpoint is kept and the early-stopping patience does
not run. Per-epoch metrics go to stdout and to `metrics.csv` in the checkpoint directory.
"""
import csv
import os
import logging
import builtins
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from flax import nnx
from jax import random
from jax.random import permutation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import DictConfig, OmegaConf

from ..utils.utils import load_obj


class MetricsLogger:
    """Collects metric dicts per step and keeps `path` (a CSV) up to date."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.rows = {}

    def log(self, metrics: dict, step: int) -> None:
        row = self.rows.setdefault(int(step), {})
        for k, v in metrics.items():
            row[k] = float(v.item()) if hasattr(v, "item") else v
        if self.path is None:
            return
        keys = ["step"] + sorted({k for r in self.rows.values() for k in r})
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for s in sorted(self.rows):
                w.writerow({"step": s, **self.rows[s]})
        os.replace(tmp, self.path)


def validated_beta_min_frac(x) -> float:
    """Where the beta warm-up starts, as a fraction of the final beta, in [0, 1)."""
    v = float(x)
    if not 0.0 <= v < 1.0:
        raise ValueError(f"beta_min_frac must be in [0, 1), got {v}")
    return v


class VAETrainerModule:
    def __init__(self, model: nnx.Module, config: DictConfig, logger=None):
        self.model = model
        self.config = config
        self.optim_config = config["optimizer"]
        self.train_config = config["training"]
        self.loss_config = config["loss"]
        self.batch_size = config["datamodule"]["batch_size"]
        self.grad_clipping_config = config.get("grad_clipping", None)
        self.scheduler_config = config.get("scheduler", None)
        self.logger = logger if logger is not None else MetricsLogger()

        self.seed = self.train_config["seed"]
        self.early_stopping_patience = self.train_config["early_stopping_patience"]
        self.early_stopping_grace = self.train_config["early_stopping_grace"]
        self.eval_every = self.train_config["eval_every"]
        self.model_checkpoint_path = self.train_config["checkpoint"]

        self.create_functions()
        self._log_param_count()

        self.loss_func = load_obj(self.loss_config["class_name"])(
            **self.loss_config.get("params", {})
        )
        loss_params = self.loss_config.get("params", {})
        self._max_beta = loss_params.get("beta", 1.0)
        self._beta_warmup_epochs = loss_params.get("beta_warmup_epochs", 0)
        self._beta_min_frac = validated_beta_min_frac(
            loss_params.get("beta_min_frac", 0.0))

        with open(os.path.join(self.model_checkpoint_path, "config.yml"), "w") as f:
            OmegaConf.save(self.config, f)

    def _log_param_count(self):
        params = nnx.state(self.model, nnx.Param, ...)[0]
        n = builtins.sum(x.size for x in jax.tree_util.tree_leaves(params))
        logging.info(f"VAE parameters: {n / 1e6:.2f}M")

    def _build_lr_schedule(self, num_steps: int):
        sched_cfg = self.scheduler_config
        cls = load_obj(sched_cfg["class_name"])
        params = dict(sched_cfg.get("params", {}))
        return cls(**params)

    def init_optimizer(self, num_steps: int):
        transforms = []
        if self.grad_clipping_config is not None:
            transforms.append(
                load_obj(self.grad_clipping_config["class_name"])(
                    **self.grad_clipping_config.get("params", {})
                )
            )
        lr_schedule = self._build_lr_schedule(num_steps)
        transforms.append(
            optax.inject_hyperparams(load_obj(self.optim_config["class_name"]))(
                lr_schedule, **self.optim_config.get("params", {})
            )
        )
        self.optimizer = nnx.Optimizer(self.model, optax.chain(*transforms))

    def _current_beta(self, epoch: int) -> jnp.ndarray:
        """beta ramps linearly from beta_min_frac * beta to beta over beta_warmup_epochs."""
        if self._beta_warmup_epochs > 0:
            scale = min(1.0, epoch / self._beta_warmup_epochs)
            scale = self._beta_min_frac + (1.0 - self._beta_min_frac) * scale
        else:
            scale = 1.0
        return jnp.float32(scale * self._max_beta)

    def train_model(self, train_array, val_array, num_epochs: int):
        max_steps = num_epochs * train_array.shape[0] // self.batch_size
        self.init_optimizer(max_steps)

        best_val_elbo = -jnp.inf
        best_val_epoch = 0
        best_ckpt = ""

        for epoch in range(num_epochs):
            beta = self._current_beta(epoch)
            with self.model.stats.time({"time": {"train_epoch"}}, print=0):
                self.model.train()
                stats = self.train_epoch(self.model, self.optimizer, train_array, beta)
                self.model.stats({"train": jax.tree.map(lambda x: x.item(), stats)})

            lr_idx = 1 if self.grad_clipping_config is not None else 0
            stats["lr"] = self.optimizer.opt_state[lr_idx].hyperparams["learning_rate"]
            stats["beta"] = beta

            bad = [k for k, v in stats.items()
                   if hasattr(v, "item") and not np.isfinite(float(v))]
            if bad:
                raise FloatingPointError(
                    f"epoch {epoch}: {', '.join(sorted(bad))} went non-finite. "
                    f"beta={beta:.4g}, lr={float(stats['lr']):.3g}.")

            print(
                *self.model.stats.latest(
                    f"VAE {epoch:03d} {self.model.stats['time']['train_epoch'][-1]:.2f}s",
                    {"train": "*"},
                )
            )
            self.logger.log({"train_" + k: v for k, v in stats.items()}, step=epoch)

            if epoch % self.eval_every == 0:
                # The selection ELBO is evaluated at the final beta, not the warm-up beta.
                val_stats = self.eval_model(val_array, epoch)
                val_elbo = val_stats["elbo"]
                in_warmup = epoch < self._beta_warmup_epochs
                if in_warmup or (val_elbo - best_val_elbo) > self.early_stopping_grace:
                    ckpt = f"{self.model_checkpoint_path}/ckpt_ep{epoch}_elbo{val_elbo:.2f}.pickle"
                    self.model.dump(ckpt)
                    if best_ckpt and os.path.exists(best_ckpt):
                        os.remove(best_ckpt)
                    best_ckpt = ckpt
                    # During the warm-up the latest checkpoint is kept and patience does not run.
                    best_val_elbo = val_elbo if not in_warmup else -jnp.inf
                    best_val_epoch = epoch
                elif (epoch - best_val_epoch) > self.early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch}.")
                    break

        from ngma.artifacts import write_meta
        last_path = f"{self.model_checkpoint_path}/last_checkpoint.pickle"
        self.model.dump(last_path)
        write_meta(last_path, {"kind": "vae_checkpoint", "which": "last", "epoch": int(epoch),
                               "class": f"{type(self.model).__module__}.{type(self.model).__name__}"})
        if best_ckpt and os.path.exists(best_ckpt):
            best_path = f"{self.model_checkpoint_path}/best_model.pickle"
            os.replace(best_ckpt, best_path)
            _cls = f"{type(self.model).__module__}.{type(self.model).__name__}"
            write_meta(best_path, {
                "kind": "vae_checkpoint", "which": "best", "class": _cls,
                "selected_epoch": int(best_val_epoch),
                "val_elbo": float(best_val_elbo),
                "beta_warmup_epochs": int(self._beta_warmup_epochs),
                "last_epoch": int(epoch),
            })
        self.best_val_epoch = best_val_epoch
        self.logger.log({"selected_epoch": best_val_epoch}, step=epoch)
        logging.info(
            f"Training complete after {epoch + 1} epochs; selected checkpoint is epoch "
            f"{best_val_epoch}. Checkpoints in {self.model_checkpoint_path}"
        )

    def eval_model(self, val_array, step: int, prefix: str = "val", beta=None):
        if beta is None:
            beta = jnp.float32(self._max_beta)
        with self.model.stats.time({"time": {"eval_epoch"}}, print=0):
            self.model.eval()
            stats = self.eval_epoch(self.model, val_array, beta)
            self.model.stats({"eval": jax.tree.map(lambda x: x.item(), stats)})

        print(
            *self.model.stats.latest(
                f"VAE {step:03d}", {"eval": "*"}
            )
        )
        self.logger.log({f"{prefix}_{k}": v for k, v in stats.items()}, step=step)
        self._save_reconstructions(val_array, prefix)
        return stats

    def _save_reconstructions(self, images, prefix):
        """Eight random images and their reconstructions, written to
        `reconstructions_<prefix>.png` in the checkpoint directory (overwritten at every
        evaluation). The two random draws it makes, the images and the reparametrised codes,
        are part of the model's random stream, so training noise depends on them."""
        imgs = jnp.clip(random.choice(self.model.rngs.preview(), images, shape=(8,)), 0.0, 1.0)
        recons, _, _ = self.model(imgs)
        recons = jnp.clip(recons, 0.0, 1.0)
        show = (lambda x: x) if imgs.shape[-1] == 3 else (lambda x: x[..., 0])
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(np.asarray(show(imgs[i])), cmap="gray")
            axes[1, i].imshow(np.asarray(show(recons[i])), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(self.model_checkpoint_path, f"reconstructions_{prefix}.png"))
        plt.close(fig)

    def restore_best(self):
        """Load best_model.pickle back into the model, so the final evaluation describes the
        checkpoint the later stages use."""
        path = os.path.join(self.model_checkpoint_path, "best_model.pickle")
        if os.path.exists(path):
            self.model.load(path)
            self.model.eval()
            logging.info(f"Restored selected checkpoint for final evaluation: {path}")
        else:
            logging.warning(
                f"No best_model.pickle in {self.model_checkpoint_path}; the final numbers "
                "describe the last epoch, not the selected checkpoint."
            )
        return self.model

    def create_functions(self):
        raise NotImplementedError


class VAETrainer(VAETrainerModule):
    def create_functions(self):
        batch_size = self.batch_size

        @nnx.jit
        def train_epoch(model, optimizer, train, beta):
            n_full = train.shape[0] // batch_size
            perm = permutation(model.rngs.permut(), n_full * batch_size)
            batches = train[perm].reshape(n_full, batch_size, *train.shape[1:])

            grad_loss_fn = nnx.value_and_grad(self.loss_func, has_aux=True)

            def step(carry, batch):
                model, opt = carry
                (loss, (_, stats)), grads = grad_loss_fn(model, batch, beta)
                opt.update(grads)
                return (model, opt), stats

            (model_out, opt_out), stats_stack = nnx.scan(step, in_axes=(nnx.Carry, 0))(
                (model, optimizer), batches
            )
            # Write the scanned carry back explicitly, in case the transform returns fresh objects.
            if model_out is not model:
                nnx.update(model, nnx.state(model_out))
            if opt_out is not optimizer:
                nnx.update(optimizer, nnx.state(opt_out))
            return jax.tree.map(lambda x: x.mean(), stats_stack)

        @nnx.jit
        def eval_epoch(model, valid, beta):
            n_full = valid.shape[0] // batch_size
            perm = permutation(model.rngs.permut(), n_full * batch_size)
            batches = valid[perm].reshape(n_full, batch_size, *valid.shape[1:])

            def step(model, batch):
                _, (_, stats) = self.loss_func(model, batch, beta)
                return model, stats

            _, stats_stack = nnx.scan(step, in_axes=(nnx.Carry, 0))(model, batches)
            return jax.tree.map(lambda x: x.mean(), stats_stack)

        self.train_epoch = train_epoch
        self.eval_epoch = eval_epoch
