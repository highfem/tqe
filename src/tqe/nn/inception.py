"""Inception-ResNet classifier backbone."""

import flax.linen as nn
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class InceptionResNetConfig:
  """Hyperparameters for :class:`MiniInceptionResNet`."""

  num_classes: int
  scale_a: float = 0.17
  scale_b: float = 0.10
  scale_c: float = 0.20


class ConvLnRelu(nn.Module):
  """Convolution followed by layer normalization and ReLU."""

  out_channels: int
  kernel_size: tuple
  strides: tuple = (1, 1)
  padding: str = "SAME"  # "SAME" or "VALID"
  use_bias: bool = False

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(
      features=self.out_channels,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=self.use_bias,
    )(x)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    return x


class InceptionResNetA(nn.Module):
  """Inception-ResNet-A block with three parallel branches."""

  scale: float = 0.17

  @nn.compact
  def __call__(self, x):
    *_, C = x.shape
    branch0 = ConvLnRelu(32, (1, 1))(x)
    branch1 = ConvLnRelu(32, (1, 1))(x)
    branch1 = ConvLnRelu(32, (3, 3))(branch1)
    branch2 = ConvLnRelu(32, (1, 1))(x)
    branch2 = ConvLnRelu(48, (3, 3))(branch2)
    branch2 = ConvLnRelu(64, (3, 3))(branch2)
    x_res = jnp.concatenate([branch0, branch1, branch2], axis=-1)
    x_res = nn.Conv(C, kernel_size=(1, 1), use_bias=True)(x_res)
    return nn.relu(x + self.scale * x_res)


class InceptionResNetB(nn.Module):
  """Inception-ResNet-B block with asymmetric convolutions."""

  scale: float = 0.17

  @nn.compact
  def __call__(self, x):
    *_, C = x.shape
    branch0 = ConvLnRelu(192, (1, 1))(x)
    branch1 = ConvLnRelu(128, (1, 1))(x)
    branch1 = ConvLnRelu(160, (1, 7))(branch1)
    branch1 = ConvLnRelu(192, (7, 1))(branch1)
    x_res = jnp.concatenate([branch0, branch1], axis=-1)
    x_res = nn.Conv(C, kernel_size=(1, 1), use_bias=True)(x_res)
    return nn.relu(x + self.scale * x_res)


class InceptionResNetC(nn.Module):
  """Inception-ResNet-C block with factorized 3x3 convolutions."""

  scale: float = 0.20

  @nn.compact
  def __call__(self, x):
    *_, C = x.shape
    branch0 = ConvLnRelu(192, (1, 1))(x)
    branch1 = ConvLnRelu(192, (1, 1))(x)
    branch1 = ConvLnRelu(224, (1, 3))(branch1)
    branch1 = ConvLnRelu(256, (3, 1))(branch1)
    x_res = jnp.concatenate([branch0, branch1], axis=-1)
    x_res = nn.Conv(C, kernel_size=(1, 1), use_bias=True)(x_res)
    return nn.relu(x + self.scale * x_res)


class ReductionA(nn.Module):
  """Spatial reduction block that halves resolution via conv and max-pool."""

  @nn.compact
  def __call__(self, x):
    x0 = ConvLnRelu(192, (1, 1))(x)
    x0 = ConvLnRelu(224, (3, 3), strides=(2, 2), padding="VALID")(x0)

    x1 = ConvLnRelu(192, (1, 1))(x)
    x1 = ConvLnRelu(192, (3, 3), strides=(2, 2), padding="VALID")(x1)

    x2 = nn.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")

    return jnp.concatenate([x0, x1, x2], axis=-1)


class ReductionB(nn.Module):
  """Spatial reduction block with four parallel downsample paths."""

  @nn.compact
  def __call__(self, x):
    x0 = ConvLnRelu(256, (1, 1))(x)
    x0 = ConvLnRelu(384, (3, 3), strides=(2, 2), padding="VALID")(x0)

    x1 = ConvLnRelu(256, (1, 1))(x)
    x1 = ConvLnRelu(288, (3, 3), strides=(2, 2), padding="VALID")(x1)

    x2 = ConvLnRelu(256, (1, 1))(x)
    x2 = ConvLnRelu(288, (3, 3))(x2)
    x2 = ConvLnRelu(320, (3, 3), strides=(2, 2), padding="VALID")(x2)

    x3 = nn.max_pool(x, (3, 3), strides=(2, 2), padding="VALID")
    return jnp.concatenate([x0, x1, x2, x3], axis=-1)


class Stem(nn.Module):
  """Initial convolutional stem that reduces spatial resolution."""

  @nn.compact
  def __call__(self, x):
    x = ConvLnRelu(32, (3, 3))(x)
    x = ConvLnRelu(32, (3, 3))(x)
    x = ConvLnRelu(64, (3, 3))(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
    x = ConvLnRelu(80, (1, 1))(x)
    x = ConvLnRelu(192, (3, 3))(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
    return x


class MiniInceptionResNet(nn.Module):
  """Compact Inception-ResNet classifier returning logits and embeddings."""

  config: InceptionResNetConfig

  @nn.compact
  def __call__(self, inputs, is_training):
    x = Stem()(inputs)
    for _ in range(5):
      x = InceptionResNetA(scale=self.config.scale_a)(x)
    x = ReductionA()(x)
    for _ in range(5):
      x = InceptionResNetB(scale=self.config.scale_b)(x)
    x = ReductionB()(x)
    for _ in range(3):
      x = InceptionResNetC(scale=self.config.scale_c)(x)
    embedding = jnp.mean(x, axis=(1, 2))
    x = nn.Dropout(0.2)(embedding, deterministic=not is_training)
    x = nn.Dense(self.config.num_classes)(x)
    return x, embedding


def make_model(config):
  """Build a Mini Inception-ResNet classifier from experiment config.

  Args:
    config: Config object with an ``nn`` sub-config.

  Returns:
    Initialized :class:`MiniInceptionResNet` module.
  """
  return MiniInceptionResNet(InceptionResNetConfig(**config.nn.to_dict()))
