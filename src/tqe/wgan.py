"""WGAN-GP; returns ``((step_fn, eval_fn), sample_fn, None)``."""

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import optax
from jax import numpy as jnp
from jax import random as jr

from tqe._types import ObjectiveFns, TrainFns


@dataclasses.dataclass
class WGANConfig:
  """WGAN-GP objective configuration.

  Attributes:
    lamb: Gradient penalty coefficient in the critic loss.
    n_critic_steps: Update the generator once every this many critic steps.
  """

  lamb: float = 10.0
  n_critic_steps: int = 1

  def __post_init__(self):
    if self.lamb <= 0:
      raise ValueError(f"lamb must be > 0, got {self.lamb}")
    if self.n_critic_steps < 1:
      raise ValueError(
        f"n_critic_steps must be >= 1, got {self.n_critic_steps}"
      )


class WGANState(NamedTuple):
  """Joint training state for a WGAN critic and generator.

  Attributes:
    critic: Critic ``TrainState``.
    generator: Generator ``TrainState`` (must carry ``ema_params``).
    step: Global training step counter; controls generator update frequency.
  """

  critic: Any
  generator: Any
  step: Any = 0


def wgan(
  generator_fn: Callable,
  critic_fn: Callable,
  config: WGANConfig,
  ema_rate: float,
) -> ObjectiveFns:
  """Build WGAN-GP training functions.

  Args:
    generator_fn: Flax generator ``apply``; returns synthetic data.
    critic_fn: Flax critic ``apply``; returns critic scores.
    config: WGAN configuration.
    ema_rate: EMA decay rate for generator parameters.

  Returns:
    ``((step_fn, eval_fn), sample_fn, None)``.
    ``step_fn(rng_key, state, batch) -> (metrics, new_state)`` updates the
    critic every step and the generator every ``config.n_critic_steps`` steps.
    ``eval_fn`` and ``sample_fn`` share the same ``(rng_key, state, …)`` seam
    as every other objective factory.  ``state`` is a :class:`WGANState`.
    Batch keys: ``inputs``, ``context``, optional ``condition``.
    Losses are pmap-reduced.
  """

  def _apply_critic(params, rng_key, inputs, context, condition, is_training):
    return critic_fn(
      variables={"params": params},
      rngs={"dropout": rng_key},
      inputs=inputs,
      context=context,
      condition=condition,
      is_training=is_training,
    )

  def _apply_generator(params, rng_key, context, condition, is_training):
    drop_key, sample_key = jr.split(rng_key)
    return generator_fn(
      variables={"params": params},
      rngs={"dropout": drop_key, "sample": sample_key},
      context=context,
      condition=condition,
      is_training=is_training,
    )

  @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
  @partial(jax.grad, argnums=2)
  def _critic_forward(params, rng_key, inputs, context, condition):
    value = _apply_critic(
      params, rng_key, inputs[None], context[None], condition[None], False
    )
    return jnp.mean(value)

  def _compute_critic(rng_key, state: WGANState, batch):
    k1_key, k2_key, _ = jr.split(rng_key, 3)
    inputs, context, condition = (
      batch["inputs"],
      batch["context"],
      batch.get("condition"),
    )
    synthetic = _apply_generator(
      state.generator.params, k1_key, context, condition, False
    )
    synthetic_crit = _apply_critic(
      state.critic.params, k2_key, synthetic, context, condition, False
    )
    inputs_crit = _apply_critic(
      state.critic.params, k2_key, inputs, context, condition, False
    )
    return {
      "synthetic_crit": jnp.mean(synthetic_crit),
      "inputs_critic": jnp.mean(inputs_crit),
    }

  def critic_loss_fn(params, generator_params, rng_key, batch, is_training):
    inputs, context, condition = (
      batch["inputs"],
      batch["context"],
      batch.get("condition"),
    )
    sample_key, c1_key, c2_key, rng_key = jr.split(rng_key, 4)
    synthetic_data = _apply_generator(
      generator_params, sample_key, context, condition, False
    )
    synthetic_data = jax.lax.stop_gradient(synthetic_data)

    preds_synthetic = _apply_critic(
      params, c1_key, synthetic_data, context, condition, is_training
    )
    preds_synthetic_loss = jnp.mean(
      preds_synthetic, axis=tuple(range(1, preds_synthetic.ndim))
    )
    del preds_synthetic
    preds_inputs = _apply_critic(
      params, c2_key, inputs, context, condition, is_training
    )
    preds_inputs_loss = jnp.mean(
      preds_inputs, axis=tuple(range(1, preds_inputs.ndim))
    )
    del preds_inputs

    sample_key, rng_key = jr.split(rng_key)
    new_shape = (1,) * (inputs.ndim - 1)
    epsilon = jr.uniform(sample_key, shape=(inputs.shape[0], *new_shape))
    data_mix = inputs * epsilon + synthetic_data * (1 - epsilon)

    grad_key = jr.split(rng_key, data_mix.shape[0])
    gradients = _critic_forward(params, grad_key, data_mix, context, condition)
    gradients = gradients.reshape((gradients.shape[0], -1))
    grad_norm = jnp.linalg.norm(gradients, axis=1)
    grad_penalty = (grad_norm - 1) ** 2

    loss = preds_synthetic_loss - preds_inputs_loss + config.lamb * grad_penalty
    return jnp.mean(loss)

  def generator_loss_fn(params, critic_params, rng_key, batch, is_training):
    context, condition = batch["context"], batch.get("condition")
    gen_key, crit_key = jr.split(rng_key)
    synthetic_data = _apply_generator(
      params, gen_key, context, condition, is_training
    )
    preds = _apply_critic(
      critic_params,
      crit_key,
      synthetic_data,
      context,
      condition,
      False,
    )
    return -jnp.mean(preds)

  @jax.jit
  def step_fn(rng_key, state: WGANState, batch):
    cl_key, gl_key = jr.split(rng_key)

    crit_grad_fn = jax.value_and_grad(critic_loss_fn)
    loss_c, crit_grads = crit_grad_fn(
      state.critic.params, state.generator.params, cl_key, batch, True
    )
    loss_c = jax.lax.pmean(loss_c, axis_name="batch")
    crit_grads = jax.lax.pmean(crit_grads, axis_name="batch")
    new_critic = state.critic.apply_gradients(grads=crit_grads)

    # Generator update frequency is config.n_critic_steps. Both branches of
    # lax.cond are traced, so the skip branch pays the cost of computing
    # gradients without applying them — acceptable for research-scale models.
    def _update_generator(operand):
      gen_state, critic_params, key = operand
      gen_grad_fn = jax.value_and_grad(generator_loss_fn)
      loss_g, gen_grads = gen_grad_fn(
        gen_state.params, critic_params, key, batch, True
      )
      loss_g = jax.lax.pmean(loss_g, axis_name="batch")
      gen_grads = jax.lax.pmean(gen_grads, axis_name="batch")
      new_gen = gen_state.apply_gradients(grads=gen_grads)
      new_ema = optax.incremental_update(
        new_gen.params, new_gen.ema_params, step_size=1.0 - ema_rate
      )
      return loss_g, new_gen.replace(ema_params=new_ema)

    def _skip_generator(operand):
      gen_state, critic_params, key = operand
      loss_g = generator_loss_fn(
        gen_state.params, critic_params, key, batch, False
      )
      loss_g = jax.lax.pmean(loss_g, axis_name="batch")
      return loss_g, gen_state

    loss_g, new_generator = jax.lax.cond(
      state.step % config.n_critic_steps == 0,
      _update_generator,
      _skip_generator,
      (state.generator, new_critic.params, gl_key),
    )

    new_state = WGANState(
      critic=new_critic, generator=new_generator, step=state.step + 1
    )
    return {"loss_c": loss_c, "loss_g": loss_g}, new_state

  @jax.jit
  def eval_fn(rng_key, state: WGANState, batch):
    loss_c_key, loss_g_key, rng_key = jr.split(rng_key, 3)
    loss_c = critic_loss_fn(
      state.critic.params, state.generator.params, loss_c_key, batch, False
    )
    loss_g = generator_loss_fn(
      state.generator.params, state.critic.params, loss_g_key, batch, False
    )
    loss_c = jax.lax.pmean(loss_c, axis_name="batch")
    loss_g = jax.lax.pmean(loss_g, axis_name="batch")
    crit = _compute_critic(rng_key, state, batch)
    return {"loss_c": loss_c, "loss_g": loss_g} | crit

  def sample_fn(rng_key, state: WGANState, context, condition=None, **kwargs):
    return _apply_generator(
      state.generator.ema_params, rng_key, context, condition, False
    )

  return ObjectiveFns(TrainFns(step_fn, eval_fn), sample_fn, None)
