"""Convolutional VAE encoder and decoder for waveform latents."""

from collections.abc import Sequence

import chex
import jax
from einops import rearrange
from flax import linen as nn
from flax import struct
from jax import numpy as jnp
from jax import random as jr


@struct.dataclass
class VAEConfig:
  """Hyperparameters shared by :class:`Encoder` and :class:`Decoder`."""

  n_channels: int
  n_out_channels: int
  channel_multipliers: Sequence[int]

  n_resnet_blocks: int
  attention_resolutions: Sequence[int]
  downsampling_strides: Sequence[Sequence[int]]
  n_attention_heads: int = 1

  kernel_size: int = 3
  dropout_rate: float = 0.1
  use_conv_in_resize: bool = True
  n_groups: int = 32


class _ResidualBlock(nn.Module):
  n_out_channels: int
  dropout_rate: float
  kernel_size: int
  n_groups: int

  @nn.compact
  def __call__(self, inputs, is_training):
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


class _DotProductAttention(nn.Module):
  n_heads: int = 1

  @nn.compact
  def __call__(self, inputs):
    B, H, W, C = inputs.shape
    chex.assert_equal(C % (3 * self.n_heads), 0)
    q, k, v = jnp.split(inputs, 3, axis=3)
    return _CrossDotProductAttention(self.n_heads)(q, k, v)


class _CrossDotProductAttention(nn.Module):
  n_heads: int = 1

  @nn.compact
  def __call__(self, q, k, v):
    B, H, W, C = q.shape
    outputs = nn.attention.dot_product_attention(
      rearrange(q, "b h w (heads c) -> b (h w) heads c", heads=self.n_heads),
      rearrange(k, "b h w (heads c) -> b (h w) heads c", heads=self.n_heads),
      rearrange(v, "b h w (heads c) -> b (h w) heads c", heads=self.n_heads),
    )
    outputs = rearrange(
      outputs,
      "b (h w) heads c -> b h w (heads c)",
      heads=self.n_heads,
      h=H,
      w=W,
    )
    return outputs


class _AttentionBlock(nn.Module):
  n_heads: int
  n_groups: int

  @nn.compact
  def __call__(self, inputs, is_training, **kwargs):
    hidden = inputs
    hidden = nn.GroupNorm(num_groups=min(self.n_groups, hidden.shape[-1] // 4))(
      hidden
    )
    # input projection (replacing the MLP on conventional attention)
    hidden = nn.Conv(
      inputs.shape[-1] * 3,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding="SAME",
    )(hidden)
    # attention, we don't push through linear layers since we have the
    # convolution above outputting 3 times the layers which we use as
    # k, q, v
    hidden = _DotProductAttention(self.n_heads)(hidden)
    # output projection (replacing the MLP on conventional attention)
    outputs = nn.Conv(
      inputs.shape[-1],
      kernel_size=(1, 1),
      strides=(1, 1),
      padding="SAME",
      kernel_init=nn.initializers.zeros,
    )(hidden)
    return outputs + inputs


class _Downsample(nn.Module):
  use_conv: bool
  kernel_size: int = 3
  strides: int = (2, 2)

  @nn.compact
  def __call__(self, inputs, is_training):
    if self.use_conv:
      outputs = nn.Conv(
        inputs.shape[-1],
        kernel_size=(self.kernel_size, self.kernel_size),
        strides=self.strides,
        padding="SAME",
      )(inputs)
    else:
      outputs = nn.avg_pool(
        inputs,
        window_shape=self.strides,
        strides=self.strides,
      )
    return outputs


class _Upsample(nn.Module):
  use_conv: bool
  n_out_channels: int | None = None
  kernel_size: int = 3
  strides: int = (2, 2)

  @nn.compact
  def __call__(self, inputs, is_training):
    B, H, W, C = inputs.shape
    outputs = jax.image.resize(
      inputs,
      (B, H * self.strides[0], W * self.strides[1], C),
      method="nearest",
    )
    if self.use_conv:
      n_out_channels = self.n_out_channels or inputs.shape[-1]
      outputs = nn.Conv(
        n_out_channels,
        kernel_size=(self.kernel_size, self.kernel_size),
        strides=(1, 1),
        padding="SAME",
      )(outputs)
    return outputs


class Encoder(nn.Module):
  """VAE encoder: U-Net left half with middle block, outputs latent parameters."""

  config: VAEConfig

  def setup(self):
    chex.assert_equal(
      len(self.config.channel_multipliers) - 1,
      len(self.config.downsampling_strides),
    )

  @nn.compact
  def __call__(self, inputs, is_training, **kwargs):
    # the input is assumed to be channel last (as is the convention in Flax)
    # B, H, W, C = inputs.shape
    hidden = inputs
    # lift data
    hidden = nn.Conv(
      self.config.n_channels * self.config.channel_multipliers[0],
      kernel_size=(self.config.kernel_size, self.config.kernel_size),
      strides=(1, 1),
      padding="SAME",
    )(hidden)

    # left block of UNet
    for level, channel_mult in enumerate(self.config.channel_multipliers):
      n_outchannels = channel_mult * self.config.n_channels
      for _ in range(self.config.n_resnet_blocks):
        hidden = _ResidualBlock(
          n_out_channels=n_outchannels,
          dropout_rate=self.config.dropout_rate,
          kernel_size=self.config.kernel_size,
          n_groups=self.config.n_groups,
        )(hidden, is_training)
        if hidden.shape[1] in self.config.attention_resolutions:
          hidden = _AttentionBlock(
            n_heads=self.config.n_attention_heads, n_groups=self.config.n_groups
          )(hidden, is_training)
      if level != len(self.config.channel_multipliers) - 1:
        hidden = _Downsample(
          use_conv=self.config.use_conv_in_resize,
          kernel_size=self.config.kernel_size,
          strides=self.config.downsampling_strides[level],
        )(hidden, is_training)

    # middle block of UNet
    for i in range(2):
      hidden = _ResidualBlock(
        n_out_channels=hidden.shape[-1],
        dropout_rate=self.config.dropout_rate,
        kernel_size=self.config.kernel_size,
        n_groups=self.config.n_groups,
      )(hidden, is_training)
      if i < 2:
        hidden = _AttentionBlock(
          n_heads=self.config.n_attention_heads, n_groups=self.config.n_groups
        )(hidden, is_training)

    hidden = nn.GroupNorm(
      num_groups=min(self.config.n_groups, hidden.shape[-1] // 4)
    )(hidden)
    hidden = nn.swish(hidden)
    outputs = nn.Conv(
      self.config.n_out_channels,
      kernel_size=(self.config.kernel_size, self.config.kernel_size),
      strides=(1, 1),
      padding="SAME",
      kernel_init=nn.initializers.zeros,
    )(hidden)

    return outputs


class Decoder(nn.Module):
  """VAE decoder: U-Net right half with middle block, reconstructs from latent."""

  config: VAEConfig

  @nn.compact
  def __call__(self, inputs, is_training, **kwargs):
    # the input is assumed to be channel last (as is the convention in Flax)
    # B, H, W, C = inputs.shape
    hidden = inputs
    # lift data
    hidden = nn.Conv(
      self.config.n_channels * self.config.channel_multipliers[-1],
      kernel_size=(self.config.kernel_size, self.config.kernel_size),
      strides=(1, 1),
      padding="SAME",
    )(hidden)

    # middle block of UNet
    for i in range(2):
      hidden = _ResidualBlock(
        n_out_channels=hidden.shape[-1],
        dropout_rate=self.config.dropout_rate,
        kernel_size=self.config.kernel_size,
        n_groups=self.config.n_groups,
      )(hidden, is_training)
      if i < 2:
        hidden = _AttentionBlock(
          n_heads=self.config.n_attention_heads, n_groups=self.config.n_groups
        )(hidden, is_training)

    for level, channel_mult in reversed(
      list(enumerate(self.config.channel_multipliers))
    ):
      n_outchannels = channel_mult * self.config.n_channels
      for idx in range(self.config.n_resnet_blocks + 1):
        hidden = _ResidualBlock(
          n_out_channels=n_outchannels,
          dropout_rate=self.config.dropout_rate,
          kernel_size=self.config.kernel_size,
          n_groups=self.config.n_groups,
        )(hidden, is_training)
        if hidden.shape[1] in self.config.attention_resolutions:
          hidden = _AttentionBlock(
            n_heads=self.config.n_attention_heads, n_groups=self.config.n_groups
          )(hidden, is_training)
        if level and idx == self.config.n_resnet_blocks:
          n_outchannels = channel_mult * self.config.n_channels
          hidden = _Upsample(
            use_conv=self.config.use_conv_in_resize,
            kernel_size=self.config.kernel_size,
            strides=self.config.downsampling_strides[level - 1],
          )(hidden, is_training)

    hidden = nn.GroupNorm(
      num_groups=min(self.config.n_groups, hidden.shape[-1] // 4)
    )(hidden)
    hidden = nn.swish(hidden)
    outputs = nn.Conv(
      self.config.n_out_channels,
      kernel_size=(self.config.kernel_size, self.config.kernel_size),
      strides=(1, 1),
      padding="SAME",
      kernel_init=nn.initializers.zeros,
    )(hidden)

    return outputs


class VAE(nn.Module):
  """Variational autoencoder with reparameterized latent sampling."""

  encoder_config: VAEConfig
  decoder_config: VAEConfig

  def setup(self):
    self._encoder = Encoder(self.encoder_config, name="encoder")
    self._decoder = Decoder(self.decoder_config, name="decoder")

  def __call__(self, inputs, is_training):
    z, (shift, log_scale) = self.encode(inputs, is_training)
    outputs = self.decode(z, is_training)
    chex.assert_equal_shape([inputs, outputs])
    return outputs, (z, shift, log_scale)

  def encode(self, inputs, is_training):
    rng_key = self.make_rng("sample")
    shift_log_scale = self._encoder(inputs, is_training)
    shift, log_scale = jnp.split(shift_log_scale, 2, axis=-1)
    z = shift + jnp.exp(log_scale) * jr.normal(rng_key, log_scale.shape)
    return z, (shift, log_scale)

  def decode(self, inputs, is_training):
    return self._decoder(inputs, is_training)

  def sample(self, z, is_training=False):
    return self.decode(z, is_training)


def make_model(config):
  """Build a VAE from experiment config with separate encoder/decoder settings.

  Args:
    config: Config object with ``base_model``, ``encoder``, and ``decoder``
      sub-configs.

  Returns:
    Initialized :class:`VAE` module.
  """
  encoder_config = config.base_model.to_dict() | config.encoder.to_dict()
  decoder_config = config.base_model.to_dict() | config.decoder.to_dict()
  return VAE(VAEConfig(**encoder_config), VAEConfig(**decoder_config))
