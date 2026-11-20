"""The prior and the observation noise shared by both VAE architectures.

The generative model is decided by `PriorOpts`: a standard normal prior or a VampPrior
(a mixture of the encoder's posteriors at K learnable pseudo-inputs), and optionally a
learned scalar observation standard deviation. An architecture supplies two things:

    _encoder_moments(x) -> (mu, logvar)   its encoder path, for a batch
    _image_shape        -> (H, W, C)      the shape of a pseudo-input
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax.random import split


#: Bounds on every encoder log-variance, applied where it is produced, so that neither
#: exp(-logvar) nor exp(logvar) can overflow in the mixture density or the KL.
LOGVAR_MIN, LOGVAR_MAX = -30.0, 5.0


@dataclass
class PriorOpts:
    """Generative-model options shared by every VAE."""
    prior: str = "standard"      # "standard" N(0,I) | "vamp" VampPrior mixture
    vamp_K: int = 500            # learnable pseudo-inputs (vamp only)
    learn_sigma: bool = False    # scalar observation std for loss rec_type=l2_sigma
    log_sigma_min: float = None
    """Floor for the learned observation log-std, or None for no floor."""
    vamp_tied_logvar_min: float = None
    """Floor for the mixture components' log-variances (vamp only), or None for no floor.
    The components take their variances from the encoder applied to the pseudo-inputs."""


class PriorMixin:
    """VampPrior and learned-sigma parameters, shared by both VAE architectures."""

    _image_shape: tuple = ()

    def _init_prior_params(self, rngs: nnx.Rngs) -> None:
        """Create the optional log_sigma / pseudo_inputs parameters. Called from __init__."""
        if self.opts.learn_sigma:
            self.log_sigma = nnx.Param(jnp.zeros(()))
        if self.opts.prior == "vamp":
            self.pseudo_inputs = nnx.Param(
                jax.random.uniform(rngs.pseudo(),
                                   (self.opts.vamp_K, *self._image_shape))
            )

    def init_pseudo_inputs(self, images: jnp.ndarray, noise_std: float = 0.01) -> None:
        """Initialise the pseudo-inputs from K random training images plus small noise."""
        assert self.opts.prior == "vamp", "init_pseudo_inputs requires prior='vamp'"
        K = self.opts.vamp_K
        key_idx, key_noise = split(self.rngs.pseudo(), 2)
        idx = jax.random.choice(key_idx, images.shape[0], (K,), replace=images.shape[0] < K)
        u = images[idx].reshape(K, *self._image_shape)
        u = u + noise_std * jax.random.normal(key_noise, u.shape)
        self.pseudo_inputs.value = u.astype(self.pseudo_inputs.value.dtype)

    def vamp_pseudo_posteriors(self):
        """The mixture components (mu_k, logvar_k), each (K, z_dim): the encoder's posteriors
        at the pseudo-inputs, with the optional floor on the log-variances."""
        mu_k, logvar_k = self._encoder_moments(self.pseudo_inputs.value)
        _lo = getattr(self.opts, "vamp_tied_logvar_min", None)
        if _lo is not None:
            logvar_k = jnp.maximum(logvar_k, float(_lo))
        return mu_k, logvar_k

    def log_vamp_prior(self, z: jnp.ndarray) -> jnp.ndarray:
        """log p_vamp(z) = logsumexp_k log N(z; mu_k, diag exp(logvar_k)) - log K, (batch,)."""
        mu_k, logvar_k = self.vamp_pseudo_posteriors()           # (K, d)
        logvar_k = jnp.clip(logvar_k, LOGVAR_MIN, LOGVAR_MAX)
        diff = z[:, None, :] - mu_k[None]                        # (B, K, d)
        log_2pi = jnp.log(2.0 * jnp.pi)
        log_comp = -0.5 * jnp.sum(
            log_2pi + logvar_k[None] + diff**2 * jnp.exp(-logvar_k)[None], axis=-1
        )                                                        # (B, K)
        return jax.nn.logsumexp(log_comp, axis=1) - jnp.log(self.opts.vamp_K)
