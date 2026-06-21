"""Consistency distillation; returns ``((step_fn, eval_fn), sample_fn, None)``."""

import dataclasses
from collections.abc import Callable

import chex
import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random as jr

from tqe._types import ObjectiveFns, TrainFns


@dataclasses.dataclass
class EDMConsistencyDistillationConfig:
  """EDM consistency distillation configuration.

  Attributes:
    n_sampling_steps: Number of sampler steps during inference.
    consistency_condition: Consistency parameterization name (``"edm_style"``).
    sigma_min: Minimum noise level in the sampling schedule.
    sigma_max: Maximum noise level in the sampling schedule.
    rho: Exponent controlling the sigma schedule curvature.
    sigma_data: Data standard deviation used in preconditioning scalings.
    P_mean: Mean of the log-normal training noise distribution.
    P_std: Standard deviation of the log-normal training noise distribution.
    S_churn: Stochastic churn magnitude for sampler noise injection.
    S_min: Minimum sigma at which churn is applied.
    S_max: Maximum sigma at which churn is applied.
    S_noise: Multiplicative factor on injected churn noise.
    k: Curriculum decay shaping constant for the ``t -> r`` mapping.
    q: Base for exponential curriculum stage decay.
    b: Sigmoid steepness for curriculum adjustment.
    c: Offset in the curriculum adjustment.
  """

  n_sampling_steps: int
  consistency_condition: str
  sigma_min: float = 0.002
  sigma_max: float = 80.0
  rho: float = 7.0
  sigma_data: float = 0.5
  P_mean: float = -1.2
  P_std: float = 1.2
  S_churn: float = 40
  S_min: float = 0.05
  S_max: float = 50
  S_noise: float = 1.003

  k: float = 8.0
  q: float = 2.0
  b: float = 1.0
  c: float = 0.0

  consistency_function: Callable = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    if self.n_sampling_steps < 1:
      raise ValueError(
        f"n_sampling_steps must be >= 1, got {self.n_sampling_steps}"
      )
    if self.sigma_min <= 0:
      raise ValueError(f"sigma_min must be > 0, got {self.sigma_min}")
    if self.sigma_min >= self.sigma_max:
      raise ValueError(
        f"sigma_min ({self.sigma_min}) must be < sigma_max ({self.sigma_max})"
      )
    if self.rho <= 0:
      raise ValueError(f"rho must be > 0, got {self.rho}")
    if self.sigma_data <= 0:
      raise ValueError(f"sigma_data must be > 0, got {self.sigma_data}")
    if self.P_std <= 0:
      raise ValueError(f"P_std must be > 0, got {self.P_std}")

    if self.consistency_condition == "edm_style":
      var_data = self.sigma_data**2
      sigma_data = self.sigma_data

      def fn(inputs, outputs, times):
        new_shape = (-1,) + tuple(
          np.ones(inputs.ndim - 1, dtype=np.int32).tolist()
        )
        skip = (var_data / (times**2 + var_data)).reshape(new_shape)
        out = (times * sigma_data / jnp.sqrt(times**2 + var_data)).reshape(
          new_shape
        )
        return inputs * skip + outputs * out

      self.consistency_function = fn
    else:
      raise ValueError(
        f"unknown consistency_condition {self.consistency_condition!r};"
        f" expected 'edm_style'"
      )

  def sigma(self, eps):
    return jnp.exp(eps * self.P_std + self.P_mean)

  def loss_weight(self, sigma):
    return (jnp.square(sigma) + jnp.square(self.sigma_data)) / jnp.square(
      sigma * self.sigma_data
    )

  def in_scaling(self, sigma):
    return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

  def noise_conditioning(self, sigma):
    return 0.25 * jnp.log(sigma)

  def sampling_sigmas(self, num_steps):
    rho_inv = 1 / self.rho
    sigmas_max = self.sigma_max**rho_inv
    sigmas_min = self.sigma_min**rho_inv
    step_idxs = jnp.arange(num_steps, dtype=jnp.float32)
    sigmas = sigmas_max + step_idxs / (num_steps - 1) * (
      sigmas_min - sigmas_max
    )
    sigmas = sigmas**self.rho
    return jnp.concatenate([sigmas, jnp.zeros_like(sigmas[:1])])

  def sigma_hat(self, sigma, num_steps):
    gamma = (
      jnp.minimum(self.S_churn / num_steps, 2**0.5 - 1)
      if self.S_min <= sigma <= self.S_max
      else 0
    )
    return sigma + gamma * sigma


def diffusion_consistency_distillation(
  model_fn: Callable,
  config: EDMConsistencyDistillationConfig,
  ema_rate: float,
) -> ObjectiveFns:
  """Build consistency distillation training functions.

  Args:
    model_fn: Flax ``apply``; returns raw network outputs for consistency
      preconditioning.
    config: Consistency distillation configuration.
    ema_rate: EMA decay rate; ``step_fn`` updates ``state.ema_params``.

  Returns:
    ``((step_fn, eval_fn), sample_fn, None)``. ``step_fn`` / ``eval_fn`` use batch keys
    ``inputs``, ``context``, and optional ``condition``, plus curriculum stage
    ``n_stage``. Loss and gradients are pmap-reduced.
  """

  def ratio(t, n_stage):
    adj = 1 + config.k * jax.nn.sigmoid(-config.b * t)
    decay = 1 / config.q ** (n_stage + 1)
    return 1 - decay * adj

  def t_to_r(t, n_stage):
    r = t * ratio(t, n_stage)
    return jnp.clip(r, min=0, max=None)

  def _denoise(rng_key, params, inputs, sigma, context, condition, is_training):
    new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
    inputs_t = inputs * config.in_scaling(sigma).reshape(new_shape)
    noise_cond = config.noise_conditioning(sigma)
    outputs = model_fn(
      variables={"params": params},
      rngs={"dropout": rng_key},
      inputs=inputs_t,
      context=context,
      times=noise_cond,
      condition=condition,
      is_training=is_training,
    )
    return config.consistency_function(inputs, outputs, sigma)

  def loss_fn(params, rng_key, batch, is_training, n_stage):
    inputs, context = batch["inputs"], batch["context"]
    condition = batch.get("condition")
    chex.assert_rank(inputs, 4)
    new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())

    epsilon_key, rng_key = jr.split(rng_key)
    epsilon = jr.normal(epsilon_key, (inputs.shape[0],))
    t_sigma = config.sigma(epsilon)
    r_sigma = t_to_r(t_sigma, n_stage)
    noise_key, rng_key = jr.split(rng_key)
    noise = jr.normal(noise_key, inputs.shape)
    t_noise = noise * t_sigma.reshape(new_shape)
    r_noise = noise * r_sigma.reshape(new_shape)

    denoise_key, rng_key = jr.split(rng_key)
    v_t = _denoise(
      denoise_key,
      params,
      inputs=inputs + t_noise,
      sigma=t_sigma,
      context=context,
      condition=condition,
      is_training=is_training,
    )
    v_r = _denoise(
      denoise_key,
      params,
      inputs=inputs + r_noise,
      sigma=r_sigma,
      context=context,
      condition=condition,
      is_training=is_training,
    )
    v_r = jax.lax.stop_gradient(v_r)
    v_r = jnp.where(r_noise > 0.0, v_r, inputs)

    loss = jnp.square(v_r - v_t)
    loss = loss.sum(axis=tuple(range(1, loss.ndim))) / jnp.squeeze(
      t_sigma - r_sigma
    )
    return loss.mean()

  @jax.jit
  def step_fn(rng_key, state, batch, n_stage):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, rng_key, batch, True, n_stage)
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
  def eval_fn(rng_key, state, batch, n_stage):
    loss = loss_fn(state.params, rng_key, batch, False, n_stage)
    loss = jax.lax.pmean(loss, axis_name="batch")
    return {"loss": loss}

  def sample_fn(rng_key, state, context, condition=None, **kwargs):
    n = context.shape[0]

    noise_key, rng_key = jr.split(rng_key)
    sigmas = config.sampling_sigmas(config.n_sampling_steps)
    noise = jr.normal(noise_key, context.shape) * sigmas[0]

    samples = noise
    for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
      sample_key, rng_key = jr.split(rng_key)
      samples = _denoise(
        sample_key,
        state.ema_params,
        inputs=samples,
        sigma=jnp.repeat(sigma, n),
        context=context,
        condition=condition,
        is_training=False,
      )
      sample_key, rng_key = jr.split(rng_key)
      samples = samples + sigma_next * jr.normal(sample_key, context.shape)

    return samples

  return ObjectiveFns(TrainFns(step_fn, eval_fn), sample_fn, None)
