"""U-Net GAN (pix2pix-style) generator and PatchGAN critic."""

from functools import partial

import chex
import jax
from flax import linen as nn
from flax import struct
from jax import numpy as jnp
from jax import random as jr

from tqe.nn.vit import ConditionEmbedding


@struct.dataclass
class UNetGANGeneratorConfig:
  """Hyperparameters for :class:`UNetGenerator`."""

  n_out_channels: int
  n_channels: int
  n_embedding_dimension: int
  append_noise_in_generator: bool = False
  dropout: float = 0.0
  kernel_size: tuple[int, int] = (3, 3)


@struct.dataclass
class PatchGANGCriticConfig:
  """Hyperparameters for :class:`PatchGANCritic`."""

  n_out_channels: int
  n_channels: int
  n_embedding_dimension: int
  dropout: float = 0.0
  kernel_size: tuple[int, int] = (3, 3)
  use_patch_gan: bool = True


def _downsample(inputs):
  n, h, w, c = inputs.shape
  return jax.image.resize(
    inputs, shape=(n, h // 2, w // 2, c), method="bilinear"
  )


def _upsample(inputs):
  n, h, w, c = inputs.shape
  return jax.image.resize(inputs, shape=(n, h * 2, w * 2, c), method="bilinear")


class DownConv(nn.Module):
  """Downsampling convolution block with layer norm and optional activation."""

  n_channels: int
  kernel_size: tuple[int, int] = (3, 3)
  activate: bool = True
  dropout: float = 0.0

  @nn.compact
  def __call__(self, inputs, *args):
    conv_fn = partial(
      nn.Conv,
      kernel_size=self.kernel_size,
      strides=(1, 1),
      padding="SAME",
    )

    return nn.Sequential(
      [
        nn.LayerNorm(),
        _downsample,
        conv_fn(self.n_channels),
        lambda x: nn.leaky_relu(x) if self.activate else x,
        lambda x: nn.Dropout(self.dropout, deterministic=False)(x),
      ]
    )(inputs)


class UpConv(nn.Module):
  """Upsampling convolution block with layer norm and optional activation."""

  n_channels: int
  kernel_size: tuple[int, int] = (3, 3)
  activate: bool = True
  dropout: float = 0.0

  @nn.compact
  def __call__(self, inputs, *args):
    conv_fn = partial(
      nn.Conv,
      kernel_size=self.kernel_size,
      strides=(1, 1),
      padding="SAME",
    )

    return nn.Sequential(
      [
        nn.LayerNorm(),
        _upsample,
        conv_fn(self.n_channels),
        lambda x: nn.leaky_relu(x) if self.activate else x,
        lambda x: nn.Dropout(self.dropout, deterministic=False)(x),
      ]
    )(inputs)


class UpConditionConv(nn.Module):
  """Upsampling block with additive condition bias."""

  n_channels: int
  kernel_size: tuple[int, int] = (3, 3)
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, condition):
    x = UpConv(
      self.n_channels, self.kernel_size, activate=False, dropout=self.dropout
    )(x)
    condition = nn.Sequential([jax.nn.silu, nn.Dense(x.shape[-1])])(condition)
    x = jax.nn.leaky_relu(x + condition[:, None, None, :])
    return x


class DownConditionConv(nn.Module):
  """Downsampling block with additive condition bias."""

  n_channels: int
  kernel_size: tuple[int, int] = (3, 3)
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, condition):
    x = DownConv(
      self.n_channels, self.kernel_size, activate=False, dropout=self.dropout
    )(x)
    condition = nn.Sequential([jax.nn.silu, nn.Dense(x.shape[-1])])(condition)
    x = jax.nn.leaky_relu(x + condition[:, None, None, :])
    return x


class UNetGenerator(nn.Module):
  """U-Net generator with skip connections for pix2pix translation."""

  config: UNetGANGeneratorConfig

  @nn.compact
  def __call__(self, context, condition, is_training=True):
    if self.config.append_noise_in_generator:
      noise = jr.normal(self.make_rng("sample"), context.shape)
      context = jnp.concatenate([context, noise], axis=-1)
    condition = ConditionEmbedding(self.config.n_embedding_dimension)(condition)

    hidden = nn.Conv(
      self.config.n_channels,
      kernel_size=self.config.kernel_size,
      strides=(1, 1),
      padding="SAME",
    )(context)
    hidden = jax.nn.leaky_relu(hidden)

    downs = [
      DownConv(self.config.n_channels * 2),  # 16
      DownConditionConv(self.config.n_channels * 2),  # 8
      DownConv(self.config.n_channels * 4, dropout=self.config.dropout),  # 4
      DownConv(self.config.n_channels * 4, dropout=self.config.dropout),  # 2
    ]
    ups = [
      UpConv(self.config.n_channels * 4, dropout=self.config.dropout),  # 4
      UpConditionConv(
        self.config.n_channels * 4, dropout=self.config.dropout
      ),  # 8
      UpConv(self.config.n_channels * 2),  # 16
      UpConv(self.config.n_channels * 2),  # 32
    ]

    skips = []
    for down in downs:
      skips.append(hidden)
      hidden = down(hidden, condition)
    for up in ups:
      sk = skips.pop()
      hidden = up(hidden, condition)
      hidden = jnp.concatenate([hidden, sk], axis=-1)

    outputs = nn.Sequential(
      [
        nn.Conv(
          self.config.n_out_channels,
          kernel_size=self.config.kernel_size,
          strides=(1, 1),
          padding="SAME",
          kernel_init=nn.initializers.normal(stddev=0.02),
        )
      ]
    )(hidden)

    chex.assert_axis_dimension(outputs, 0, context.shape[0])
    chex.assert_axis_dimension(outputs, 1, context.shape[1])
    chex.assert_axis_dimension(outputs, 2, context.shape[2])
    if self.config.append_noise_in_generator:
      chex.assert_axis_dimension(outputs, 3, context.shape[3] // 2)
    else:
      chex.assert_axis_dimension(outputs, 3, context.shape[3])
    return outputs


class PatchGANCritic(nn.Module):
  """PatchGAN or fully-connected critic for pix2pix adversarial training."""

  config: PatchGANGCriticConfig

  @nn.compact
  def __call__(self, inputs, context, condition, is_training=True):
    hidden = jnp.concatenate([inputs, context], axis=-1)
    condition = ConditionEmbedding(self.config.n_embedding_dimension)(condition)

    hidden = nn.Conv(
      self.config.n_channels,
      kernel_size=self.config.kernel_size,
      strides=(1, 1),
      padding="SAME",
    )(hidden)
    hidden = jax.nn.leaky_relu(hidden)

    downs = [
      DownConv(self.config.n_channels * 2),  # 16
      DownConditionConv(self.config.n_channels * 2),  # 8
      DownConv(self.config.n_channels * 4),  # 4
      DownConv(self.config.n_channels * 4),  # 2
    ]

    for down in downs:
      hidden = down(hidden, condition)

    if self.config.use_patch_gan:
      outputs = nn.Sequential(
        [
          lambda x: jnp.pad(
            x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant"
          ),
          nn.Conv(
            self.config.n_channels * 4,
            kernel_size=self.config.kernel_size,
            strides=(1, 1),
            padding="VALID",
          ),
          nn.LayerNorm(),
          jax.nn.leaky_relu,
          lambda x: jnp.pad(
            x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant"
          ),
          nn.Conv(
            1,
            kernel_size=self.config.kernel_size,
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.normal(stddev=0.02),
          ),
        ]
      )(hidden)
    else:
      outputs = nn.Sequential(
        [
          lambda x: jnp.reshape(x, (x.shape[0], -1)),
          nn.Dense(1, kernel_init=nn.initializers.normal(stddev=0.02)),
        ]
      )(hidden)
    return outputs


def make_models(config):
  """Build a pix2pix U-Net generator and PatchGAN critic from experiment config.

  Args:
    config: Config object with ``generator.nn`` and ``critic.nn`` sub-configs.

  Returns:
    Tuple ``(generator, critic)`` of initialized Flax modules.
  """
  return UNetGenerator(
    UNetGANGeneratorConfig(**config.generator.nn.to_dict())
  ), PatchGANCritic(PatchGANGCriticConfig(**config.critic.nn.to_dict()))
