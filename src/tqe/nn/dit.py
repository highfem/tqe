"""Diffusion Transformer (DiT) for conditional waveform denoising."""

import chex
import jax
from einops import pack, rearrange, unpack
from flax import linen as nn
from flax import struct
from jax import numpy as jnp

from tqe.nn.timestep_embedding import timestep_embedding


@struct.dataclass
class DiTConfig:
  """Hyperparameters for :class:`DiT`."""

  # number of channels after initial lifting
  n_channels: int
  n_out_channels: int

  patch_size: int
  n_blocks: int
  n_heads: int

  n_embedding_dimension: int

  # how to condition everthing
  do_context_projections: bool
  do_context_conditioning: bool
  do_meta_conditioning: bool

  dropout_rate: float = 0.1


def get_sinusoidal_embedding_1d(length, embedding_dim):
  emb = timestep_embedding(length.reshape(-1), embedding_dim)
  return emb


def sinusoidal_init(rng, shape, dtype):
  def get_sinusoidal_embedding_2d(grid, embedding_dim):
    emb_h = get_sinusoidal_embedding_1d(grid[0], embedding_dim // 2)
    emb_w = get_sinusoidal_embedding_1d(grid[1], embedding_dim // 2)
    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb

  _, n_h_patches, n_w_patches, embedding_dim = shape
  grid_h = jnp.arange(n_h_patches, dtype=jnp.float32)
  grid_w = jnp.arange(n_w_patches, dtype=jnp.float32)
  grid = jnp.meshgrid(grid_w, grid_h)

  grid = jnp.stack(grid, axis=0)
  grid = grid.reshape([2, 1, n_w_patches, n_h_patches])
  pos_embed = get_sinusoidal_embedding_2d(grid, embedding_dim)

  return jnp.expand_dims(pos_embed, 0)  # (1, H*W, D)


def _modulate(inputs, shift, scale):
  return inputs * (1.0 + scale[:, None]) + shift[:, None]


def _layernorm_modulate(inputs, alpha, beta):
  hidden = nn.LayerNorm(use_scale=False, use_bias=False)(inputs)
  hidden = _modulate(hidden, alpha, beta)
  return hidden


def _residshiftscale_layernorm_modulate_mlp_residshiftscale(
  attns,
  inputs,
  hidden_size,
  dropout_rate,
  is_training,
  gamma,
  delta,
  epsilon,
  zeta,
):
  hidden = attns * gamma[:, None] + inputs
  intermediate = nn.LayerNorm(use_scale=False, use_bias=False)(hidden)
  intermediate = _modulate(intermediate, delta, epsilon)
  intermediate = nn.Sequential(
    [
      nn.Dense(hidden_size * 4),
      nn.gelu,
      lambda x: nn.Dropout(dropout_rate)(x, deterministic=not is_training),
      nn.Dense(hidden_size),
      lambda x: nn.Dropout(dropout_rate)(x, deterministic=not is_training),
    ]
  )(intermediate)
  outputs = intermediate * zeta[:, None] + hidden
  return outputs


class MultiHeadAttention(nn.Module):
  """MMDiT-style multi-head attention over input and optional context streams."""

  n_heads: int
  do_context_projections: bool
  do_context_conditioning: bool
  do_normalization: bool = True

  @nn.compact
  def __call__(self, inputs, context):
    if self.do_context_projections:
      chex.assert_equal(
        self.do_context_conditioning, self.do_context_projections
      )
    hidden_inputs = nn.Conv(
      3 * inputs.shape[-1],
      kernel_size=1,
      strides=1,
      padding="SAME",
    )(inputs)
    if self.do_context_projections:
      hidden_context = nn.Conv(
        3 * context.shape[-1],
        kernel_size=1,
        strides=1,
        padding="SAME",
      )(context)
    else:
      hidden_context = jnp.concatenate([context, context, context], axis=-1)

    if self.do_context_conditioning:
      all_qkvs = []
      for qkv in [hidden_inputs, hidden_context]:
        if self.do_normalization:
          q, k, v = jnp.split(qkv, 3, axis=-1)
          q, k = nn.RMSNorm()(q), nn.RMSNorm()(k)
          qkv = jnp.concatenate([q, k, v], axis=-1)
        all_qkvs.append(qkv)
    else:
      all_qkvs = [hidden_inputs]

    all_qkvs, packed_shape = pack(all_qkvs, "b * c")
    q, k, v = jnp.split(all_qkvs, 3, axis=-1)
    hidden = nn.dot_product_attention(
      rearrange(q, "b l (heads c) -> b l heads c", heads=self.n_heads),
      rearrange(k, "b l (heads c) -> b l heads c", heads=self.n_heads),
      rearrange(v, "b l (heads c) -> b l heads c", heads=self.n_heads),
    )
    hidden = rearrange(
      hidden, "b l heads c -> b l (heads c)", heads=self.n_heads
    )
    outs = unpack(hidden, packed_shape, "b * c")
    if self.do_context_projections:
      outs = tuple([nn.Dense(inputs.shape[-1])(out) for out in outs])
    else:
      outs = nn.Dense(inputs.shape[-1])(outs[0]), context
    return outs


class DiTBlock(nn.Module):
  """Single DiT block with AdaLN-modulated attention and MLP."""

  hidden_size: int
  n_heads: int
  do_context_projections: bool
  do_context_conditioning: bool
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, times, context, is_training, **kwargs):
    adaln_norm = nn.Sequential([nn.silu, nn.Dense(self.hidden_size * 12)])(
      times
    )
    mod_y, mod_c = jnp.split(adaln_norm, 2, axis=-1)
    mod_y, mod_c = jnp.split(mod_y, 6, axis=-1), jnp.split(mod_c, 6, axis=-1)

    hidden_inputs = _layernorm_modulate(inputs, mod_y[0], mod_y[1])
    hidden_context = (
      _layernorm_modulate(context, mod_c[0], mod_c[1])
      if self.do_context_projections
      else context
    )

    # do the mmdit attention
    hidden_inputs, hidden_context = MultiHeadAttention(
      self.n_heads,
      self.do_context_projections,
      self.do_context_conditioning,
    )(hidden_inputs, hidden_context)

    outputs_inputs = _residshiftscale_layernorm_modulate_mlp_residshiftscale(
      hidden_inputs,
      inputs,
      self.hidden_size,
      self.dropout_rate,
      is_training,
      *mod_y[2:],
    )
    outputs_context = (
      _residshiftscale_layernorm_modulate_mlp_residshiftscale(
        hidden_context,
        context,
        self.hidden_size,
        self.dropout_rate,
        is_training,
        *mod_c[2:],
      )
      if self.do_context_projections
      else context
    )

    return outputs_inputs, outputs_context


class DiT(nn.Module):
  """Diffusion Transformer score network with patchified input and context."""

  config: DiTConfig

  def _time_embedding(self, times):
    times = timestep_embedding(times, self.config.n_embedding_dimension)
    times = nn.Sequential(
      [
        nn.Dense(self.config.n_embedding_dimension),
        nn.silu,
        nn.Dense(self.config.n_embedding_dimension),
      ]
    )(times)
    return times

  def _condition_embedding(self, condition):
    condition = nn.Sequential(
      [
        nn.Dense(self.config.n_embedding_dimension),
        nn.silu,
        nn.Dense(self.config.n_embedding_dimension),
      ]
    )(condition)
    return condition

  def _patchify(self, inputs, idx):
    B, H, W, C = inputs.shape
    patch_size_tuple = (self.config.patch_size, self.config.patch_size)
    n_h_patches = H // self.config.patch_size
    n_w_patches = W // self.config.patch_size
    hidden = nn.Conv(
      self.config.n_channels,
      patch_size_tuple,
      patch_size_tuple,
      padding="VALID",
      kernel_init=nn.initializers.xavier_uniform(),
      name=f"patchify_{idx}",
    )(inputs)
    outputs = rearrange(
      hidden, "b h w c -> b (h w) c", h=n_h_patches, w=n_w_patches
    )
    return outputs, (n_h_patches, n_w_patches)

  def _unpatchify(self, inputs, h, w):
    B, *_ = inputs.shape
    p = q = self.config.patch_size
    hidden = jnp.reshape(
      inputs,
      (B, h, w, p, q, self.config.n_out_channels),
    )
    outputs = rearrange(
      hidden, "b h w p q c -> b (h p) (w q) c", h=h, w=w, p=p, q=q
    )
    return outputs

  def _embed(self, inputs, context, n_h_patches, n_w_patches):
    pos_emb_shape = (1, n_h_patches, n_w_patches, inputs.shape[2])
    patch_embedding = self.param(
      "patch_embedding",
      sinusoidal_init,
      pos_emb_shape,
      inputs.dtype,
    )
    patch_embedding = jax.lax.stop_gradient(patch_embedding)
    return inputs + patch_embedding, context + patch_embedding

  @nn.compact
  def __call__(self, inputs, times, context, condition=None, is_training=True):
    hidden, (n_h_patches, n_w_patches) = self._patchify(inputs, 0)
    context, _ = self._patchify(context, 1)
    hidden, context = self._embed(hidden, context, n_h_patches, n_w_patches)

    embedding = self._time_embedding(times)
    if condition is not None and self.config.do_meta_conditioning:
      condition = self._condition_embedding(condition)
      embedding = embedding + condition

    # dit blocks
    for _ in range(self.config.n_blocks):
      hidden, context = DiTBlock(
        self.config.n_channels,
        self.config.n_heads,
        do_context_projections=self.config.do_context_projections,
        do_context_conditioning=self.config.do_context_conditioning,
      )(hidden, times=embedding, context=context, is_training=is_training)

    # final layer
    embedding = nn.Sequential(
      [
        nn.silu,
        nn.Dense(self.config.n_channels * 2, kernel_init=nn.initializers.zeros),
      ]
    )(embedding)
    embedding_shift, embedding_scale = jnp.split(embedding, 2, -1)
    hidden = nn.Sequential(
      [
        nn.LayerNorm(use_scale=False, use_bias=False),
        lambda x: _modulate(x, embedding_shift, embedding_scale),
        nn.Dense(
          self.config.patch_size
          * self.config.patch_size
          * self.config.n_out_channels,
          kernel_init=nn.initializers.zeros,
        ),
      ]
    )(hidden)
    outputs = self._unpatchify(hidden, n_h_patches, n_w_patches)
    chex.assert_equal_shape([inputs, outputs])
    return outputs


def make_model(config):
  """Build a DiT score network from experiment config.

  Args:
    config: Config object convertible to :class:`DiTConfig` fields.

  Returns:
    Initialized :class:`DiT` module.
  """
  return DiT(DiTConfig(**config.to_dict()))
