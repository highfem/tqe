"""EDM diffusion; returns ``((step_fn, eval_fn), sample_fn, None)``."""

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
class EDMParameterization:
  """EDM noise schedule, scaling, and sampling configuration.

  Attributes:
    n_sampling_steps: Number of Heun solver steps during inference.
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
  """

  n_sampling_steps: int
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

  def __post_init__(self):
    if self.n_sampling_steps < 2:
      raise ValueError(
        f"n_sampling_steps must be >= 2 for the Heun sampler,"
        f" got {self.n_sampling_steps}"
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

  def sigma(self, eps):
    return jnp.exp(eps * self.P_std + self.P_mean)

  def loss_weight(self, sigma):
    return (jnp.square(sigma) + jnp.square(self.sigma_data)) / jnp.square(
      sigma * self.sigma_data
    )

  def skip_scaling(self, sigma):
    return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

  def out_scaling(self, sigma):
    return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

  def in_scaling(self, sigma):
    return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

  def noise_conditioning(self, sigma):
    return 0.25 * jnp.log(sigma)

  def sampling_sigmas(self, num_steps):
    rho_inv = 1 / self.rho
    step_idxs = jnp.arange(num_steps, dtype=jnp.float32)
    sigmas = (
      self.sigma_max**rho_inv
      + step_idxs
      / (num_steps - 1)
      * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
    ) ** self.rho
    return jnp.concatenate([sigmas, jnp.zeros_like(sigmas[:1])])

  def sigma_hat(self, sigma, num_steps):
    gamma = (
      jnp.minimum(self.S_churn / num_steps, 2**0.5 - 1)
      if self.S_min <= sigma <= self.S_max
      else 0
    )
    return sigma + gamma * sigma


def denoising_diffusion(
  model_fn: Callable, config: EDMParameterization, ema_rate: float
) -> ObjectiveFns:
  """Build EDM denoising diffusion training functions.

  Args:
    model_fn: Flax ``apply``; returns raw network outputs for EDM preconditioning.
    config: EDM parameterization.
    ema_rate: EMA decay rate; ``step_fn`` updates ``state.ema_params``.

  Returns:
    ``((step_fn, eval_fn), sample_fn, None)``. ``step_fn`` / ``eval_fn`` use batch keys
    ``inputs``, ``context``, and optional ``condition``. Loss and gradients are
    pmap-reduced.
  """

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
    skip = inputs * config.skip_scaling(sigma).reshape(new_shape)
    outputs = outputs * config.out_scaling(sigma).reshape(new_shape)
    return skip + outputs

  def loss_fn(params, rng_key, batch, is_training):
    inputs, context = batch["inputs"], batch["context"]
    condition = batch.get("condition")
    chex.assert_rank(inputs, 4)
    new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())

    epsilon_key, rng_key = jr.split(rng_key)
    epsilon = jr.normal(epsilon_key, (inputs.shape[0],))
    sigma = config.sigma(epsilon)
    noise_key, rng_key = jr.split(rng_key)
    noise = jr.normal(noise_key, inputs.shape)
    noise = noise * sigma.reshape(new_shape)

    denoise_key, rng_key = jr.split(rng_key)
    target_hat = _denoise(
      denoise_key,
      params,
      inputs=inputs + noise,
      sigma=sigma,
      context=context,
      condition=condition,
      is_training=is_training,
    )

    loss = jnp.square(inputs - target_hat)
    loss = config.loss_weight(sigma).reshape(new_shape) * loss
    return loss.mean()

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

  def sample_fn(rng_key, state, context, condition=None, **kwargs):
    del kwargs
    n = context.shape[0]
    noise_key, rng_key = jr.split(rng_key)
    sigmas = config.sampling_sigmas(config.n_sampling_steps)
    noise = jr.normal(noise_key, context.shape) * sigmas[0]

    sample_next = noise
    for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
      pred_key1, pred_key2, rng_key = jr.split(rng_key, 3)
      sample_curr = sample_next
      pred_curr = _denoise(
        pred_key1,
        state.ema_params,
        inputs=sample_curr,
        sigma=jnp.repeat(sigma, n),
        context=context,
        condition=condition,
        is_training=False,
      )
      d_cur = (sample_curr - pred_curr) / sigma
      sample_next = sample_curr + d_cur * (sigma_next - sigma)

      if i < config.n_sampling_steps - 1:
        pred_next = _denoise(
          pred_key2,
          state.ema_params,
          inputs=sample_next,
          sigma=jnp.repeat(sigma_next, n),
          context=context,
          condition=condition,
          is_training=False,
        )
        d_prime = (sample_next - pred_next) / sigma_next
        sample_next = sample_curr + (sigma_next - sigma) * (
          0.5 * d_cur + 0.5 * d_prime
        )
    return sample_next

  return ObjectiveFns(TrainFns(step_fn, eval_fn), sample_fn, None)
