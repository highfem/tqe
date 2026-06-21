"""VAE; returns ``((step_fn, eval_fn), sample_fn, reconstruct_fn)``."""

import dataclasses
from collections.abc import Callable

import jax
from jax import numpy as jnp
from jax import random as jr

from tqe._types import ObjectiveFns, TrainFns


@dataclasses.dataclass
class AutoencoderConfig:
  """VAE objective configuration.

  Attributes:
    kl_weight: Scalar multiplier on the KL divergence term.
  """

  kl_weight: float = 1e-6

  def __post_init__(self):
    if self.kl_weight < 0:
      raise ValueError(f"kl_weight must be >= 0, got {self.kl_weight}")


def kl_div(shift, log_scale):
  log_var = 2 * log_scale
  return 0.5 * jnp.sum(
    jnp.square(shift) + jnp.exp(log_var) - log_var - 1, axis=(1, 2, 3)
  )


def autoencoder(apply_fn: Callable, config: AutoencoderConfig) -> ObjectiveFns:
  """Build autoencoder training functions.

  Args:
    apply_fn: Flax ``apply``; returns ``(reconstruction, (_, shift, log_scale))``.
    config: Autoencoder configuration.

  Returns:
    ``((step_fn, eval_fn), sample_fn, reconstruct_fn)``. ``step_fn`` / ``eval_fn``
    use batch key ``inputs``. Loss and gradients are pmap-reduced.
  """

  def _apply(params, rngs, inputs, is_training):
    return apply_fn(
      {"params": params},
      rngs=rngs,
      inputs=inputs,
      is_training=is_training,
    )

  def loss_fn(params, rngs, inputs, is_training):
    output_hat, (_, shift, log_scale) = _apply(
      params, rngs, inputs, is_training
    )
    nll = jnp.sum(jnp.square(inputs - output_hat), axis=(1, 2, 3))
    weighted_kl_div = config.kl_weight * kl_div(shift, log_scale)
    return jnp.mean(nll + weighted_kl_div)

  def _split_rngs(rng_key):
    return dict(zip(["sample", "dropout"], jr.split(rng_key)))

  @jax.jit
  def step_fn(rng_key, state, batch):
    rngs = _split_rngs(rng_key)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, rngs, batch["inputs"], True)
    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    return {"loss": loss}, new_state

  @jax.jit
  def eval_fn(rng_key, state, batch):
    rngs = _split_rngs(rng_key)
    output_hat, (_, shift, log_scale) = _apply(
      state.params, rngs, batch["inputs"], False
    )
    nll = jnp.sum(jnp.square(batch["inputs"] - output_hat), axis=(1, 2, 3))
    weighted_kl_div = config.kl_weight * kl_div(shift, log_scale)
    loss = jnp.mean(nll + weighted_kl_div)
    loss = jax.lax.pmean(loss, axis_name="batch")
    return {"loss": loss}

  @jax.jit
  def reconstruct_fn(rng_key, state, batch):
    rngs = _split_rngs(rng_key)
    output_hat, (_, shift, _) = _apply(
      state.params, rngs, batch["inputs"], False
    )
    return output_hat, shift

  def sample_fn(rng_key, state, inputs, **kwargs):
    rngs = _split_rngs(rng_key)
    output_hat, _ = _apply(state.params, rngs, inputs, False)
    return output_hat

  return ObjectiveFns(TrainFns(step_fn, eval_fn), sample_fn, reconstruct_fn)
