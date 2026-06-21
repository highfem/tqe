"""Flow matching; returns ``((step_fn, eval_fn), sample_fn, None)``."""

import dataclasses
from collections.abc import Callable

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random as jr

from tqe._types import ObjectiveFns, TrainFns


def _forward_process(inputs, times, noise):
  new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
  times = times.reshape(new_shape)
  inputs_t = times * inputs + (1.0 - times) * noise
  return inputs_t


@dataclasses.dataclass
class FlowMatchingConfig:
  """Conditional flow-matching configuration.

  Attributes:
    n_sampling_steps: Number of Euler steps in the inference sampler.
    time_eps: Minimum flow time.
    time_max: Maximum flow time.
    time_embedding_scale: Rescales ``t ∈ [0, 1]`` for sinusoidal timestep
      embeddings; matches the ``T=1000`` convention from Ho et al.
  """

  n_sampling_steps: int
  time_eps: float = 1e-3
  time_max: float = 1.0
  time_embedding_scale: float = 999.0

  def __post_init__(self):
    if self.n_sampling_steps < 1:
      raise ValueError(
        f"n_sampling_steps must be >= 1, got {self.n_sampling_steps}"
      )
    if not (0.0 <= self.time_eps < self.time_max):
      raise ValueError(
        f"time_eps ({self.time_eps}) must satisfy 0 <= time_eps < time_max"
        f" ({self.time_max})"
      )
    if self.time_embedding_scale <= 0:
      raise ValueError(
        f"time_embedding_scale must be > 0, got {self.time_embedding_scale}"
      )


def flow_matching(
  model_fn: Callable, config: FlowMatchingConfig, ema_rate: float
) -> ObjectiveFns:
  """Build flow-matching loss and sampling functions.

  Args:
    model_fn: Flax ``apply``; returns predicted velocity ``vs``.
    config: Flow-matching configuration.
    ema_rate: EMA decay rate (``sample_fn`` reads ``state.ema_params``).

  Returns:
    ``((step_fn, eval_fn), sample_fn, None)``. ``step_fn`` / ``eval_fn`` use batch
    keys ``inputs``, ``context``, and optional ``condition``.
  """

  def loss_fn(params, rng_key, batch, is_training):
    inputs, context = batch["inputs"], batch["context"]
    condition = batch.get("condition")
    time_key, rng_key = jr.split(rng_key)
    times = jr.uniform(time_key, shape=(inputs.shape[0],))
    times = times * (config.time_max - config.time_eps) + config.time_eps
    noise_key, rng_key = jr.split(rng_key)
    noise = jr.normal(noise_key, inputs.shape)
    inputs_t = _forward_process(inputs, times, noise)

    vs = model_fn(
      variables={"params": params},
      rngs={"dropout": rng_key},
      inputs=inputs_t,
      context=context,
      times=times * config.time_embedding_scale,
      condition=condition,
      is_training=is_training,
    )
    target = inputs - noise
    loss = jnp.mean(jnp.square(target - vs))
    return loss

  def sample_fn(rng_key, state, context, condition=None, **kwargs):
    n = context.shape[0]
    dt = 1.0 / config.n_sampling_steps
    samples = jr.normal(rng_key, context.shape)
    for i in range(config.n_sampling_steps):
      times = i / config.n_sampling_steps
      times = times * (config.time_max - config.time_eps) + config.time_eps
      times = jnp.repeat(times, n)
      vt = state.apply_fn(
        variables={"params": state.ema_params},
        inputs=samples,
        context=context,
        times=times * config.time_embedding_scale,
        condition=condition,
        is_training=False,
      )
      samples = samples + vt * dt
    return samples

  @jax.jit
  def step_fn(rng_key, state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, rng_key, batch, True)
    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    new_ema_params = optax.incremental_update(
      new_state.params,
      new_state.ema_params,
      step_size=1.0 - ema_rate,
    )
    return {"loss": loss}, new_state.replace(ema_params=new_ema_params)

  @jax.jit
  def eval_fn(rng_key, state, batch):
    loss = loss_fn(state.params, rng_key, batch, False)
    loss = jax.lax.pmean(loss, axis_name="batch")
    return {"loss": loss}

  return ObjectiveFns(TrainFns(step_fn, eval_fn), sample_fn, None)
