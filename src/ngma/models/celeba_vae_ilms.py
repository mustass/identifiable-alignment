"""VAE for 64x64x3 images (CelebA), with an ensemble of decoders on one encoder.

Encoder: five convolutions (3 -> 128 -> 128 -> 256 -> 256 -> 256, 4x4 kernels) and a
Linear(16384 -> 256) stage before the moments. Decoder: resize-and-convolution blocks with
nearest-neighbour upsampling from an 8x8 grid, one decoder per ensemble member, built with
nnx.vmap. The interface matches ngma.models.vae.VAE.
"""

import pickle

from ngma.artifacts import save_checkpoint
from ngma.models.priors import LOGVAR_MAX, LOGVAR_MIN, PriorMixin, PriorOpts
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jax.numpy import exp
from jax.random import split

from ..utils.stats import Stats


class ResizeAndConv(nnx.Module):
    """Nearest-neighbour upsample followed by a convolution."""

    def __init__(self, in_channels, filters, kernel_size, strides, rngs):
        self.strides = strides
        self.conv = nnx.Conv(in_channels, filters, kernel_size, (1, 1), rngs=rngs)

    def __call__(self, x):
        if self.strides != (1, 1):
            x = jax.image.resize(
                x,
                (
                    x.shape[0],
                    x.shape[1] * self.strides[0],
                    x.shape[2] * self.strides[1],
                    x.shape[3],
                ),
                method="nearest",
            )
        return self.conv(x)


class VAE(PriorMixin, nnx.Module):
    @dataclass
    class DefaultOpts(PriorOpts):
        z_dim: int = 64
        num_decoders: int = 8
        dec_channels: tuple = (64, 64, 64, 32)   # widths of the four up-blocks; the head emits 3
        dec_kernel: int = 4

    class Decoder(nnx.Module):
        """8x8 -> 64x64x3 by four resize-and-convolution blocks (strides 1, 2, 2, 2) and a head."""

        def __init__(self, opts, rngs):
            self.opts = opts
            ch = tuple(int(c) for c in opts.dec_channels)
            assert len(ch) == 4, f"dec_channels must list the 4 up-blocks, got {ch}"
            k = (int(opts.dec_kernel), int(opts.dec_kernel))
            self.base_ch = ch[0]
            self.fc_dec = nnx.Sequential(
                nnx.Linear(opts.z_dim, 256, rngs=rngs),
                nnx.elu,
                nnx.Linear(256, ch[0] * 8 * 8, rngs=rngs),
                nnx.elu,
            )
            self.convs = nnx.Sequential(
                ResizeAndConv(ch[0], ch[0], kernel_size=k, strides=(1, 1), rngs=rngs),
                nnx.elu,
                ResizeAndConv(ch[0], ch[1], kernel_size=k, strides=(2, 2), rngs=rngs),
                nnx.elu,
                ResizeAndConv(ch[1], ch[2], kernel_size=k, strides=(2, 2), rngs=rngs),
                nnx.elu,
                ResizeAndConv(ch[2], ch[3], kernel_size=k, strides=(2, 2), rngs=rngs),
                nnx.elu,
                ResizeAndConv(ch[3], 3, kernel_size=k, strides=(1, 1), rngs=rngs),
            )

        def __call__(self, z):
            x = self.fc_dec(z)
            x = x.reshape(x.shape[0], 8, 8, self.base_ch)
            return self.convs(x)

    def __init__(self, opts: dict = {}, *, rngs: nnx.Rngs):
        self.stats = Stats()
        self.opts = self.DefaultOpts(**opts)
        self.rngs = rngs

        self.encoder = nnx.Sequential(
            nnx.Conv(3, 128, kernel_size=(4, 4), strides=(1, 1), rngs=rngs),
            nnx.elu,
            nnx.Conv(128, 128, kernel_size=(4, 4), strides=(2, 2), rngs=rngs),
            nnx.elu,
            nnx.Conv(128, 256, kernel_size=(4, 4), strides=(2, 2), rngs=rngs),
            nnx.elu,
            nnx.Conv(256, 256, kernel_size=(4, 4), strides=(2, 2), rngs=rngs),
            nnx.elu,
            nnx.Conv(256, 256, kernel_size=(4, 4), strides=(1, 1), rngs=rngs),
            nnx.elu,
        )
        self.enc = nnx.Linear(4 * 4 * 1024, 256, rngs=rngs)
        self.enc_mu = nnx.Linear(256, self.opts.z_dim, rngs=rngs)
        self.enc_logvar = nnx.Linear(256, self.opts.z_dim, rngs=rngs)

        rngss = nnx.vmap(lambda s: nnx.Rngs(s), in_axes=0)(
            split(rngs(), self.opts.num_decoders)
        )
        self.decoder = nnx.vmap(self.Decoder, in_axes=(None, 0))(self.opts, rngss)

        self._image_shape = (64, 64, 3)
        self._init_prior_params(rngs)

    def _encoder_moments(self, x: jnp.ndarray):
        """(mu, logvar) for a batch, with the log-variance clamped to [LOGVAR_MIN, LOGVAR_MAX]."""
        h = self.encoder(x).reshape(x.shape[0], -1)
        h = nnx.elu(self.enc(h))
        return self.enc_mu(h), jnp.clip(self.enc_logvar(h), LOGVAR_MIN, LOGVAR_MAX)

    def reparametrize(self, mu, logvar):
        std = jax.random.normal(self.rngs.reparam(), mu.shape)
        return mu + exp(0.5 * logvar) * std

    def encode(self, x, reparam: bool = False):
        z_mu, z_logvar = self._encoder_moments(x)
        if not reparam:
            return z_mu, z_mu, None
        return self.reparametrize(z_mu, z_logvar), z_mu, z_logvar

    def decode_ensemble(self, z: jnp.ndarray) -> jnp.ndarray:
        """(batch, z_dim) -> (batch, K, H*W*C): the vmap-built decoders, applied one by one."""
        graphdef, state = nnx.split(self.decoder)      # state leaves carry a leading K axis
        outs = []
        for k in range(self.opts.num_decoders):
            state_k = jax.tree_util.tree_map(lambda x: x[k], state)
            outs.append(nnx.merge(graphdef, state_k)(z))   # (batch, H, W, C)
        decoded = jnp.stack(outs, axis=1)              # (batch, K, H, W, C)
        return decoded.reshape(decoded.shape[0], self.opts.num_decoders, -1)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Mean over decoders: (batch, 64, 64, 3)."""
        flat_mean = self.decode_ensemble(z).mean(axis=1)
        return flat_mean.reshape(flat_mean.shape[0], 64, 64, 3)

    def decode_single(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decoder 0 on one latent vector: z (d,) -> (H*W*C,)."""
        return self.decode_ensemble(z[None])[0, 0]

    def decode_single_ensemble(self, z: jnp.ndarray) -> jnp.ndarray:
        """All decoders on one latent vector: z (d,) -> (K, H*W*C)."""
        return self.decode_ensemble(z[None])[0]

    def __call__(self, x, reparam: bool = True):
        z, z_mu, z_logvar = self.encode(x, reparam)
        return self.decode(z), z_mu, z_logvar

    def dump(self, path: str):
        save_checkpoint(self, path, {"opts": self.opts, "stats": self.stats, "state": nnx.state(self)})

    def load(self, path: str):
        from ..utils.stats import Stats as NgmaStats
        import flax as _flax

        _flax_ver = tuple(int(x) for x in _flax.__version__.split(".")[:2])
        _needs_vs_compat = _flax_ver >= (0, 10)

        # Checkpoints written under flax < 0.10 store VariableState as {'type', 'value'};
        # flax >= 0.10 made it an alias of Variable. A proxy loads the old format on new flax,
        # and the leaves are converted afterwards.
        class _OldVariableState:
            __slots__ = ("type", "value")
            def __setstate__(self, s):
                object.__setattr__(self, "type", s["type"])
                object.__setattr__(self, "value", s["value"])

        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if "ilms" in module:
                    if "DefaultOpts" in name:
                        return VAE.DefaultOpts
                    if "Stats" in name:
                        return NgmaStats
                if _needs_vs_compat and module == "flax.nnx.variablelib" and name == "VariableState":
                    return _OldVariableState
                if not _needs_vs_compat and module == "flax.nnx.statelib" and name == "State":
                    try:
                        from flax.nnx.graph import State as _GraphState
                    except ImportError:
                        from flax import nnx as _nnx
                        _GraphState = _nnx.State
                    return _GraphState
                return super().find_class(module, name)

        with open(path, "rb") as f:
            d = _CompatUnpickler(f).load()

        if _needs_vs_compat:
            def _upgrade(obj):
                if isinstance(obj, _OldVariableState):
                    return obj.type(obj.value)
                if isinstance(obj, dict):
                    return {k: _upgrade(v) for k, v in obj.items()}
                from flax.nnx.statelib import State as _State
                if isinstance(obj, _State):
                    return _State({k: _upgrade(v) for k, v in obj.raw_mapping.items()})
                return obj
            d["state"] = _upgrade(d["state"])

        loaded_opts = d["opts"]
        if isinstance(loaded_opts, dict):
            loaded_opts = self.DefaultOpts(**loaded_opts)
        for field_name, field in VAE.DefaultOpts.__dataclass_fields__.items():
            if not hasattr(loaded_opts, field_name):
                setattr(loaded_opts, field_name, field.default)
        self.opts = loaded_opts
        self.stats = d.get("stats", Stats())
        nnx.update(self, d["state"])
