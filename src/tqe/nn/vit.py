"""ViT blocks and condition embeddings for patchified waveforms."""

from einops import rearrange
from flax import linen as nn
from jax import numpy as jnp

from tqe.nn.dit import (
  _layernorm_modulate,
  _residshiftscale_layernorm_modulate_mlp_residshiftscale,
)


class ConditionEmbedding(nn.Module):
  """Map a low-dimensional condition vector to model hidden size."""

  n_embedding_dimension: int

  @nn.compact
  def __call__(self, condition):
    x = nn.Dense(self.n_embedding_dimension)(condition)
    x = nn.silu(x)
    x = nn.Dense(self.n_embedding_dimension)(x)
    return x


class Patchify(nn.Module):
  """Convert an image into a sequence of patch tokens via strided convolution."""

  patch_size: int
  n_channels: int

  @nn.compact
  def __call__(self, inputs):
    B, H, W, C = inputs.shape
    n_h_patches = H // self.patch_size
    n_w_patches = W // self.patch_size

    # Conv with stride = patch_size turns patches into tokens
    hidden = nn.Conv(
      features=self.n_channels,
      kernel_size=(self.patch_size, self.patch_size),
      strides=(self.patch_size, self.patch_size),
      padding="VALID",
      kernel_init=nn.initializers.xavier_uniform(),
    )(inputs)

    outputs = rearrange(hidden, "b h w c -> b (h w) c")
    return outputs, (n_h_patches, n_w_patches)


def _unpatchify(inputs, h, w, config):
  B, *_ = inputs.shape
  p = q = config.patch_size
  hidden = jnp.reshape(
    inputs,
    (B, h, w, p, q, config.n_out_channels),
  )
  outputs = rearrange(
    hidden, "b h w p q c -> b (h p) (w q) c", h=h, w=w, p=p, q=q
  )
  return outputs


class PositionalEmbedding(nn.Module):
  """Add learnable positional embeddings to a token sequence."""

  @nn.compact
  def __call__(self, inputs):
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pos_emb = self.param(
      "pos_emb",
      nn.initializers.normal(stddev=0.02),
      pos_emb_shape,
      inputs.dtype,
    )
    return inputs + pos_emb


class MultiHeadAttention(nn.Module):
  """Single-stream multi-head self-attention with a 1x1 conv QKV projection."""

  n_heads: int
  hidden_size: int | None = None

  @nn.compact
  def __call__(self, inputs, mask=None):
    out_size = self.hidden_size | inputs.shape[-1]
    hidden_inputs = nn.Conv(
      3 * inputs.shape[-1],
      kernel_size=1,
      strides=1,
      padding="SAME",
    )(inputs)

    q, k, v = jnp.split(hidden_inputs, 3, axis=-1)
    hidden = nn.dot_product_attention(
      rearrange(q, "b l (heads c) -> b l heads c", heads=self.n_heads),
      rearrange(k, "b l (heads c) -> b l heads c", heads=self.n_heads),
      rearrange(v, "b l (heads c) -> b l heads c", heads=self.n_heads),
      mask=mask,
    )
    hidden = rearrange(
      hidden, "b l heads c -> b l (heads c)", heads=self.n_heads
    )
    outs = nn.Dense(out_size)(hidden)
    return outs


class ViTBlock(nn.Module):
  """AdaLN-conditioned transformer block with self-attention and MLP."""

  n_heads: int
  hidden_size: int | None = None
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, condition, is_training, mask=None):
    mod = nn.Sequential([nn.silu, nn.Dense(self.hidden_size * 6)])(condition)
    mod = jnp.split(mod, 6, axis=-1)
    hidden = _layernorm_modulate(inputs, mod[0], mod[1])
    attn_out = MultiHeadAttention(self.n_heads, self.hidden_size)(
      hidden, mask=mask
    )
    outputs = _residshiftscale_layernorm_modulate_mlp_residshiftscale(
      attn_out,
      inputs,
      self.hidden_size,
      self.dropout_rate,
      is_training,
      *mod[2:],
    )
    return outputs
