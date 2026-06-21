"""U-Net denoising backbone with attention blocks."""

from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn
from flax import struct

from tqe.nn.vae import (
  _AttentionBlock,
  _Downsample,
  _Upsample,
)


@struct.dataclass
class UNetConfig:
  """Hyperparameters for :class:`UNet`."""

  n_channels: int
  n_out_channels: int

  channel_multipliers: Sequence[int]
  downsampling_strides: Sequence[Sequence[int]]
  n_resnet_blocks: int
  attention_resolutions: Sequence[int]
  n_embedding_multiplier: int
  n_attention_heads: int

  kernel_size: int = 3
  dropout_rate: float = 0.1
  use_conv_in_resize: bool = True
  n_groups: int = 32

  do_meta_conditioning: bool = True
  do_context_condition_with_cross_attention: bool = True
  do_context_condition_with_concat: bool = False
  n_transformer_blocks: int = 4

  pad_to: tuple | None = None


class _GaussianFourierProjection(nn.Module):
  embedding_size: int = 256
  scale: float = 16.0

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param(
      "kernel",
      jax.nn.initializers.normal(stddev=self.scale),
      (self.embedding_size // 2,),
    )
    kernel = jax.lax.stop_gradient(kernel)
    x_proj = inputs[:, None] * kernel[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class _ConditionalResidualBlock(nn.Module):
  n_out_channels: int
  dropout_rate: float
  kernel_size: int
  n_groups: int

  @nn.compact
  def __call__(self, inputs, sigma, is_training):
    hidden = inputs
    # convolution with pre-layer norm
    hidden = nn.GroupNorm(num_groups=min(self.n_groups, hidden.shape[-1] // 4))(
      hidden
    )
    hidden = nn.silu(hidden)
    hidden = nn.Conv(
      self.n_out_channels,
      kernel_size=(self.kernel_size, self.kernel_size),
      strides=(1, 1),
      padding="SAME",
    )(hidden)

    embedding = nn.Dense(self.n_out_channels)(sigma)
    hidden += embedding[:, None, None, :]

    # convolution with pre-layer norm and dropout
    hidden = nn.GroupNorm(num_groups=min(self.n_groups, hidden.shape[-1] // 4))(
      hidden
    )
    hidden = nn.silu(hidden)
    hidden = nn.Dropout(self.dropout_rate)(
      hidden, deterministic=not is_training
    )
    hidden = nn.Conv(
      self.n_out_channels,
      kernel_size=(self.kernel_size, self.kernel_size),
      strides=(1, 1),
      padding="SAME",
      kernel_init=nn.initializers.zeros,
    )(hidden)

    if inputs.shape[-1] != self.n_out_channels:
      residual = nn.Conv(
        self.n_out_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="SAME",
      )(inputs)
    else:
      residual = inputs

    return hidden + residual


class MLP(nn.Module):
  """Two-layer GELU MLP with dropout for transformer blocks."""

  dropout_rate: float

  @nn.compact
  def __call__(self, inputs, is_training):
    out_dim = inputs.shape[-1]
    x = nn.Dense(out_dim * 4)(inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not is_training)(x)
    x = nn.Dense(out_dim)(x)
    return x


class _TransformerBlock(nn.Module):
  n_heads: int
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, context, is_training):
    outputs = inputs
    hidden = nn.LayerNorm()(outputs)
    hidden = nn.SelfAttention(
      num_heads=self.n_heads, use_bias=False, dropout_rate=0.0
    )(hidden, deterministic=not is_training)
    hidden = nn.Dropout(self.dropout_rate)(
      hidden, deterministic=not is_training
    )
    outputs = outputs + hidden

    hidden = nn.LayerNorm()(outputs)
    hidden = nn.MultiHeadDotProductAttention(
      num_heads=self.n_heads, use_bias=False, dropout_rate=0.0
    )(inputs_q=hidden, inputs_kv=context, deterministic=not is_training)
    hidden = nn.Dropout(self.dropout_rate)(
      hidden, deterministic=not is_training
    )
    outputs = outputs + hidden

    hidden = nn.LayerNorm()(outputs)
    hidden = MLP(self.dropout_rate)(hidden, is_training=is_training)
    hidden = nn.Dropout(self.dropout_rate)(
      hidden, deterministic=not is_training
    )
    outputs = outputs + hidden

    return outputs


class _Transformer(nn.Module):
  n_heads: int = 1
  n_groups: int = 32
  dropout_rate: float = 0.1
  n_transformer_blocks: int = 1

  @nn.compact
  def __call__(self, inputs, context, is_training):
    B, H, W, C = inputs.shape
    hidden = inputs
    hidden = nn.GroupNorm(num_groups=min(self.n_groups, hidden.shape[-1] // 4))(
      hidden
    )
    hidden = nn.Conv(
      inputs.shape[-1],
      kernel_size=(1, 1),
      strides=(1, 1),
      padding="SAME",
    )(hidden)

    hidden = rearrange(hidden, "b h w c -> b (h w) c")
    context = rearrange(context, "b h w c -> b (h w) c")
    for _ in range(self.n_transformer_blocks):
      hidden = _TransformerBlock(self.n_heads, self.dropout_rate)(
        hidden, context, is_training=is_training
      )
    hidden = nn.LayerNorm()(hidden)
    hidden = rearrange(hidden, "b (h w) c -> b h w c", h=H, w=W)

    outputs = nn.Conv(
      inputs.shape[-1],
      kernel_size=(1, 1),
      strides=(1, 1),
      padding="SAME",
      kernel_init=nn.initializers.zeros,
    )(hidden)
    return outputs + inputs


def attention_fn(
  do_condition_with_cross_attention,
  n_heads,
  n_groups,
  n_transformer_blocks,
  dropout_rate,
):
  if do_condition_with_cross_attention:
    return _Transformer(
      n_heads=n_heads,
      n_groups=n_groups,
      n_transformer_blocks=n_transformer_blocks,
      dropout_rate=dropout_rate,
    )
  else:
    return _AttentionBlock(n_heads=n_heads, n_groups=n_groups)


class UNet(nn.Module):
  """Conditional U-Net score model with timestep and context conditioning."""

  config: UNetConfig

  def setup(self):
    chex.assert_equal(
      len(self.config.channel_multipliers) - 1,
      len(self.config.downsampling_strides),
    )
    padding_fn = lambda image: image
    unpadding_fn = lambda image, _: image
    if self.config.pad_to is not None:
      padding_fn = lambda x: jnp.pad(x, self.config.pad_to)

      def unpadding_fn(padded_image, new_shape):
        start_idxs = [
          (s1 - s2) // 2 for s1, s2 in zip(padded_image.shape, new_shape)
        ]
        return jax.lax.dynamic_slice(padded_image, start_idxs, new_shape)

    self.padding_fn = padding_fn
    self.unpadding_fn = unpadding_fn

  @nn.compact
  def __call__(
    self, inputs, times, context, condition=None, is_training=False, **kwargs
  ):
    # the input is assumed to be channel last (as is the convention in Flax)
    # B, H, W, C = inputs.shape

    hidden = self.padding_fn(inputs)
    context = self.padding_fn(context)
    if self.config.do_context_condition_with_concat:
      hidden = jnp.concatenate([hidden, context], axis=-1)

    # time embedding
    cond_embedding = nn.Sequential(
      [
        _GaussianFourierProjection(
          self.config.n_channels * self.config.n_embedding_multiplier
        ),
        nn.Dense(self.config.n_channels * self.config.n_embedding_multiplier),
        nn.silu,
        nn.Dense(self.config.n_channels * self.config.n_embedding_multiplier),
      ]
    )(times)

    # embed conditioning variables
    if condition is not None and self.config.do_meta_conditioning:
      condition = nn.Sequential(
        [
          nn.Dense(self.config.n_channels * self.config.n_embedding_multiplier),
          nn.silu,
          nn.Dense(self.config.n_channels * self.config.n_embedding_multiplier),
        ]
      )(condition)
      cond_embedding = cond_embedding + condition
    cond_embedding = nn.silu(cond_embedding)

    # lift data
    hidden = nn.Conv(
      self.config.n_channels,
      kernel_size=(self.config.kernel_size, self.config.kernel_size),
      strides=(1, 1),
      padding="SAME",
    )(hidden)

    hs = [hidden]
    # downsampling UNet blocks
    for level, channel_mult in enumerate(self.config.channel_multipliers):
      n_outchannels = channel_mult * self.config.n_channels
      for _ in range(self.config.n_resnet_blocks):
        hidden = _ConditionalResidualBlock(
          n_out_channels=n_outchannels,
          dropout_rate=self.config.dropout_rate,
          kernel_size=self.config.kernel_size,
          n_groups=self.config.n_groups,
        )(hidden, cond_embedding, is_training)
        if level in self.config.attention_resolutions:
          hidden = attention_fn(
            do_condition_with_cross_attention=self.config.do_context_condition_with_cross_attention,
            n_heads=self.config.n_attention_heads,
            n_groups=self.config.n_groups,
            n_transformer_blocks=self.config.n_transformer_blocks,
            dropout_rate=self.config.dropout_rate,
          )(inputs=hidden, context=context, is_training=is_training)
        hs.append(hidden)
      if level != len(self.config.channel_multipliers) - 1:
        hidden = _Downsample(
          use_conv=self.config.use_conv_in_resize,
          kernel_size=self.config.kernel_size,
          strides=self.config.downsampling_strides[level],
        )(hidden, is_training)
        hs.append(hidden)

    # middle UNet block
    for i in range(2):
      hidden = _ConditionalResidualBlock(
        n_out_channels=hidden.shape[-1],
        dropout_rate=self.config.dropout_rate,
        kernel_size=self.config.kernel_size,
        n_groups=self.config.n_groups,
      )(hidden, cond_embedding, is_training)
      if i < 2:
        hidden = attention_fn(
          do_condition_with_cross_attention=self.config.do_context_condition_with_cross_attention,
          n_heads=self.config.n_attention_heads,
          n_groups=self.config.n_groups,
          n_transformer_blocks=self.config.n_transformer_blocks,
          dropout_rate=self.config.dropout_rate,
        )(inputs=hidden, context=context, is_training=is_training)

    # upsampling UNet block
    for level, channel_mult in reversed(
      list(enumerate(self.config.channel_multipliers))
    ):
      n_outchannels = channel_mult * self.config.n_channels
      for idx in range(self.config.n_resnet_blocks + 1):
        hidden = jnp.concatenate([hidden, hs.pop()], axis=-1)
        hidden = _ConditionalResidualBlock(
          n_out_channels=n_outchannels,
          dropout_rate=self.config.dropout_rate,
          kernel_size=self.config.kernel_size,
          n_groups=self.config.n_groups,
        )(hidden, cond_embedding, is_training)
        if level in self.config.attention_resolutions:
          hidden = attention_fn(
            do_condition_with_cross_attention=self.config.do_context_condition_with_cross_attention,
            n_heads=self.config.n_attention_heads,
            n_groups=self.config.n_groups,
            n_transformer_blocks=self.config.n_transformer_blocks,
            dropout_rate=self.config.dropout_rate,
          )(inputs=hidden, context=context, is_training=is_training)
        if level and idx == self.config.n_resnet_blocks:
          hidden = _Upsample(
            use_conv=self.config.use_conv_in_resize,
            kernel_size=self.config.kernel_size,
            strides=self.config.downsampling_strides[level - 1],
          )(hidden, is_training)

    outputs = nn.Sequential(
      [
        nn.GroupNorm(
          num_groups=min(self.config.n_groups, hidden.shape[-1] // 4)
        ),
        nn.silu,
        nn.Conv(
          self.config.n_out_channels,
          kernel_size=(
            self.config.kernel_size,
            self.config.kernel_size,
          ),
          strides=(1, 1),
          padding="SAME",
          kernel_init=nn.initializers.zeros,
        ),
      ]
    )(hidden)
    outputs = self.unpadding_fn(outputs, inputs.shape)
    chex.assert_equal_shape([inputs, outputs])
    chex.assert_equal(len(hs), 0)
    return outputs


def make_model(config):
  """Build a conditional U-Net from experiment config.

  Args:
    config: Config object convertible to :class:`UNetConfig` fields.

  Returns:
    Initialized :class:`UNet` module.
  """
  return UNet(UNetConfig(**config.to_dict()))
