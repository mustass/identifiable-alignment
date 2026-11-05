import jax.numpy as jnp

from ngma.models.priors import LOGVAR_MAX, LOGVAR_MIN
from optax import l2_loss


class NelboLoss:
    """The beta-VAE objective, reconstruction + beta * KL, averaged over the batch.

    rec_type:
      "l2"        0.5 * sum (x - x_hat)^2
      "l2_sigma"  0.5 * sum (x - x_hat)^2 / sigma^2 + N log sigma, with a learned scalar sigma
                  (needs model.params.learn_sigma=true)

    rec_target:
      "mean"         the reconstruction error of the mean over decoders
      "per_decoder"  the mean over decoders of each decoder's own reconstruction error

    free_bits: minimum KL per latent dimension (nats); for the VampPrior it applies to the
    aggregate KL with threshold free_bits * z_dim.
    beta_warmup_epochs, beta_min_frac: read by the trainer, which passes the current beta.

    With a VampPrior the KL is the single-sample estimate log q(z|x) - log p_vamp(z) at
    z ~ q(z|x); with the standard prior it is analytic.
    """

    def __init__(self, beta: float = 1.0, rec_type: str = "l2",
                 free_bits: float = 0.0, beta_warmup_epochs: int = 0,
                 rec_target: str = "mean", beta_min_frac: float = 0.0):
        self.beta = beta
        self.free_bits = free_bits
        assert rec_type in ("l2", "l2_sigma"), f"Unknown rec_type '{rec_type}'"
        assert rec_target in ("mean", "per_decoder"), f"Unknown rec_target '{rec_target}'"
        self.rec_type = rec_type
        self.rec_target = rec_target

    def __call__(self, model, batch, beta=None):
        if beta is None:
            beta = self.beta
        vamp = getattr(getattr(model, "opts", None), "prior", "standard") == "vamp"

        z, z_mu, z_logvar = model.encode(batch, reparam=True)
        x_hat = model.decode(z)
        n_pix = batch.shape[-1] * batch.shape[-2] * batch.shape[-3]

        if self.rec_target == "per_decoder":
            outs = model.decode_ensemble(z)
            x_flat = batch.reshape(batch.shape[0], 1, -1)
            sq = (0.5 * (outs - x_flat) ** 2).sum(axis=-1).mean(axis=-1)
        else:
            sq = l2_loss(x_hat, batch).sum(axis=(-1, -2, -3))
        if self.rec_type == "l2_sigma":
            if not hasattr(model, "log_sigma"):
                raise ValueError(
                    "rec_type=l2_sigma needs the model's learned observation std: "
                    "pass model.params.learn_sigma=true")
            ls = model.log_sigma.value
            _lo = getattr(getattr(model, "opts", None), "log_sigma_min", None)
            if _lo is not None:
                ls = jnp.maximum(ls, float(_lo))
            rec_loss = jnp.exp(-2.0 * ls) * sq + n_pix * ls
        else:
            rec_loss = sq

        stats_extra = {
            "z_logvar_min": z_logvar.min(),
            "z_logvar_mean": z_logvar.mean(),
        }

        if vamp:
            log_2pi = jnp.log(2.0 * jnp.pi)
            _lv = jnp.clip(z_logvar, LOGVAR_MIN, LOGVAR_MAX)
            log_q = -0.5 * jnp.sum(
                log_2pi + _lv + (z - z_mu) ** 2 * jnp.exp(-_lv), axis=-1
            )
            log_p = model.log_vamp_prior(z)
            stats_extra["log_p_vamp"] = log_p.mean()
            kl_loss = jnp.maximum(log_q - log_p, self.free_bits * z_mu.shape[-1])
        else:
            kl_per_dim = -0.5 * (1.0 + z_logvar - z_mu**2 - jnp.exp(z_logvar))
            kl_loss = jnp.sum(jnp.maximum(kl_per_dim, self.free_bits), axis=-1)

        loss = rec_loss + beta * kl_loss

        stats = {
            "elbo": -loss.mean(),
            "kl_loss": kl_loss.mean(),
            "rec_loss": rec_loss.mean(),
        }
        mse = 2.0 * sq.mean() / n_pix
        stats["psnr"] = -10.0 * jnp.log10(jnp.maximum(mse, 1e-12))
        if self.rec_type == "l2_sigma":
            stats["sigma"] = jnp.exp(model.log_sigma.value)
        stats.update(stats_extra)
        return loss.mean(), ((x_hat, z_mu, z_logvar), stats)
