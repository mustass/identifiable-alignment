"""VAE for 28x28 grayscale images (MNIST, FashionMNIST), with an ensemble of decoders on one
encoder.

The encoder reduces the image to a 2x2x256 bottleneck with four stride-2 convolutions. Each
decoder mirrors it with four stride-2 residual upsampling blocks to 32x32 and a bilinear
resize to the target size.
"""

import jax
import jax.numpy as jnp
from jax.random import split
from jax.numpy import exp
from flax import nnx
from dataclasses import dataclass
import pickle

from ngma.artifacts import save_checkpoint
from ngma.models.priors import LOGVAR_MAX, LOGVAR_MIN, PriorMixin, PriorOpts

# flax < 0.10 needs nnx.List to traverse a list of submodules; flax >= 0.10 traverses plain lists.
_ModuleList = getattr(nnx, "List", list)

from ..utils.stats import Stats


class ResizeAndConv(nnx.Module):
    """Bilinear upsample followed by a convolution."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, rngs):
        self.stride = stride
        self.conv = nnx.Conv(in_ch, out_ch, kernel_size, padding=padding, rngs=rngs)

    def __call__(self, x):
        if self.stride != 1:
            h, w = x.shape[1] * self.stride, x.shape[2] * self.stride
            x = jax.image.resize(x, (x.shape[0], h, w, x.shape[3]), method="linear")
        return self.conv(x)


class ResidualUpBlock(nnx.Module):
    """Upsampling block with a bilinear skip connection."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, rngs):
        self.conv = ResizeAndConv(in_ch, out_ch, kernel_size, stride, padding, rngs)
        self.skip = nnx.Conv(in_ch, out_ch, (1, 1), padding=0, rngs=rngs)
        self.stride = stride

    def __call__(self, x):
        h = self.conv(x)
        if self.stride != 1:
            sh, sw = x.shape[1] * self.stride, x.shape[2] * self.stride
            s = jax.image.resize(x, (x.shape[0], sh, sw, x.shape[3]), method="linear")
        else:
            s = x
        return h + self.skip(s)


class VAE(PriorMixin, nnx.Module):
    @dataclass
    class DefaultOpts(PriorOpts):
        z_dim: int = 16
        num_decoders: int = 1
        in_channels: int = 1
        image_h: int = 28
        image_w: int = 28

    class _Decoder(nnx.Module):
        """One decoder: z (batch, d) -> x (batch, image_h, image_w, in_channels)."""

        def __init__(self, z_dim: int, in_channels: int, image_h: int, image_w: int, rngs: nnx.Rngs):
            self.image_h = image_h
            self.image_w = image_w
            self.in_channels = in_channels
            self.fc = nnx.Linear(z_dim, 256 * 4, rngs=rngs)
            self.up1 = ResidualUpBlock(256, 128, (3, 3), stride=2, padding=1, rngs=rngs)
            self.up2 = ResidualUpBlock(128,  64, (3, 3), stride=2, padding=1, rngs=rngs)
            self.up3 = ResidualUpBlock( 64,  32, (3, 3), stride=2, padding=1, rngs=rngs)
            self.up4 = ResidualUpBlock( 32, in_channels, (3, 3), stride=2, padding=1, rngs=rngs)

        def __call__(self, z):
            h = nnx.elu(self.fc(z))
            h = h.reshape(h.shape[0], 2, 2, 256)
            h = nnx.elu(self.up1(h))
            h = nnx.elu(self.up2(h))
            h = nnx.elu(self.up3(h))
            h = nnx.sigmoid(self.up4(h))
            return jax.image.resize(
                h, (h.shape[0], self.image_h, self.image_w, self.in_channels),
                method="linear"
            )

    def __init__(self, opts: dict = {}, *, rngs: nnx.Rngs):
        self.stats = Stats()
        self.opts = self.DefaultOpts(**opts)
        self.rngs = rngs

        self._build()

    def _build(self, wrap_decoders=None):
        # When rebuilding (checkpoint load), drop the previous submodules so the module graph
        # matches `self.opts`.
        for name in ("encoder", "enc_mu", "enc_logvar", "_single_decoder", "_decoders",
                     "pseudo_inputs", "log_sigma"):
            if hasattr(self, name):
                delattr(self, name)

        rngs = self.rngs
        z  = self.opts.z_dim
        ic = self.opts.in_channels

        ih = self.opts.image_h
        iw = self.opts.image_w

        import math
        enc_h = math.ceil(math.ceil(math.ceil(math.ceil(ih / 2) / 2) / 2) / 2)
        enc_w = math.ceil(math.ceil(math.ceil(math.ceil(iw / 2) / 2) / 2) / 2)
        enc_flat = enc_h * enc_w * 256
        self.encoder = nnx.Sequential(
            nnx.Conv(ic,  32,  (3, 3), strides=2, padding=1, rngs=rngs),
            nnx.elu,
            nnx.Conv(32,  64,  (3, 3), strides=2, padding=1, rngs=rngs),
            nnx.elu,
            nnx.Conv(64,  128, (3, 3), strides=2, padding=1, rngs=rngs),
            nnx.elu,
            nnx.Conv(128, 256, (3, 3), strides=2, padding=1, rngs=rngs),
            nnx.elu,
        )
        self.enc_mu = nnx.Linear(enc_flat, z, rngs=rngs)
        self.enc_logvar = nnx.Linear(enc_flat, z, rngs=rngs)

        K = self.opts.num_decoders
        keys = split(rngs(), K)

        self.decoders = _ModuleList([
            self._Decoder(z, ic, ih, iw, nnx.Rngs(key))
            for key in keys
        ])

        self._image_shape = (ih, iw, ic)
        self._init_prior_params(rngs)

    def _encoder_moments(self, x: jnp.ndarray):
        """(mu, logvar) for a batch, with the log-variance clamped to [LOGVAR_MIN, LOGVAR_MAX]."""
        h = self.encoder(x).reshape(x.shape[0], -1)
        return self.enc_mu(h), jnp.clip(self.enc_logvar(h), LOGVAR_MIN, LOGVAR_MAX)

    def reparametrize(self, mu, logvar):
        eps = jax.random.normal(self.rngs.reparam(), mu.shape)
        return mu + exp(0.5 * logvar) * eps

    def encode(self, x, reparam: bool = False):
        z_mu, z_logvar = self._encoder_moments(x)
        if not reparam:
            return z_mu, z_mu, None
        return self.reparametrize(z_mu, z_logvar), z_mu, z_logvar

    def decode_ensemble(self, z: jnp.ndarray) -> jnp.ndarray:
        """Every decoder on the same batch: (batch, n_decoders, H*W*C)."""
        outs = jnp.stack([decoder(z) for decoder in self.decoders], axis=1)
        batch_size = outs.shape[0]
        K = outs.shape[1]
        return outs.reshape(batch_size, K, -1)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Mean over decoders: (batch, image_h, image_w, in_channels)."""
        flat_mean = self.decode_ensemble(z).mean(axis=1)
        ih, iw, ic = self.opts.image_h, self.opts.image_w, self.opts.in_channels
        return flat_mean.reshape(flat_mean.shape[0], ih, iw, ic)

    def decode_single(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decoder 0 on one latent vector: z (d,) -> x_flat (H*W*C,)."""
        return self.decode_ensemble(z[None])[0, 0]

    def decode_single_ensemble(self, z: jnp.ndarray) -> jnp.ndarray:
        """All decoders on one latent vector: z (d,) -> (n_decoders, H*W*C)."""
        return self.decode_ensemble(z[None])[0]

    def __call__(self, x, reparam: bool = True):
        z, z_mu, z_logvar = self.encode(x, reparam)
        return self.decode(z), z_mu, z_logvar

    def dump(self, path: str):
        save_checkpoint(self, path, {"opts": self.opts, "stats": self.stats, "state": nnx.state(self)})

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        loaded = d["opts"]
        if isinstance(loaded, dict):
            loaded = self.DefaultOpts(**loaded)
        for field_name, field in VAE.DefaultOpts.__dataclass_fields__.items():
            if not hasattr(loaded, field_name):
                setattr(loaded, field_name, field.default)
        self.opts = loaded
        self.stats = d["stats"]

        state = d["state"]
        self._build(wrap_decoders=hasattr(nnx, "data"))
        try:
            nnx.update(self, state)
        except Exception:
            if self.opts.num_decoders > 1 and hasattr(nnx, "data"):
                self._build(wrap_decoders=False)
                nnx.update(self, state)
            else:
                raise
