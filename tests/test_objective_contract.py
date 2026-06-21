"""Guardrails for tqe objective factory return shape and one pmap step."""

from typing import Any

import jax
import optax
import pytest
from flax import core, jax_utils, struct
from flax import linen as fnn
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import random as jr

from tqe.autoencoder import AutoencoderConfig, autoencoder
from tqe.classifier import ClassifierConfig, classifier
from tqe.denoising_diffusion import EDMParameterization, denoising_diffusion
from tqe.diffusion_consistency_distillation import (
  EDMConsistencyDistillationConfig,
  diffusion_consistency_distillation,
)
from tqe.flow_matching import FlowMatchingConfig, flow_matching
from tqe.wgan import WGANConfig, WGANState, wgan


class EMATrainState(TrainState):
  ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


class _TinyScore(fnn.Module):
  @fnn.compact
  def __call__(self, inputs, context, times, condition, is_training):
    del times, condition, is_training
    h = jnp.concatenate([inputs, context], axis=-1)
    return fnn.Dense(inputs.shape[-1])(h)


class _TinyVAE(fnn.Module):
  channels: int

  @fnn.compact
  def __call__(self, inputs, is_training):
    del is_training
    h = fnn.Conv(4, (3, 3), padding="SAME")(inputs)
    shift = fnn.Conv(self.channels, (3, 3), padding="SAME")(h)
    log_scale = fnn.Conv(self.channels, (3, 3), padding="SAME")(h)
    z = shift + jnp.exp(log_scale) * jr.normal(
      self.make_rng("sample"), log_scale.shape
    )
    output_hat = fnn.Conv(self.channels, (3, 3), padding="SAME")(z)
    return output_hat, (z, shift, log_scale)


class _TinyClassifier(fnn.Module):
  n_classes: int

  @fnn.compact
  def __call__(self, inputs, is_training):
    del is_training
    logits = fnn.Dense(self.n_classes)(inputs.reshape((inputs.shape[0], -1)))
    return logits, None


class _TinyGenerator(fnn.Module):
  @fnn.compact
  def __call__(self, context, condition, is_training):
    del condition, is_training
    return fnn.Dense(context.shape[-1])(context)


class _TinyCritic(fnn.Module):
  @fnn.compact
  def __call__(self, inputs, context, condition, is_training):
    del context, condition, is_training
    return fnn.Dense(1)(inputs)


def _assert_factory_contract(
  result,
  *,
  expect_sample_fn: bool,
  expect_extra: bool,
):
  assert isinstance(result, tuple) and len(result) == 3
  step_eval, sample_fn, extra = result
  assert isinstance(step_eval, tuple) and len(step_eval) == 2
  step_fn, eval_fn = step_eval
  assert callable(step_fn)
  assert callable(eval_fn)
  if expect_sample_fn:
    assert callable(sample_fn)
  else:
    assert sample_fn is None
  if expect_extra:
    assert callable(extra)
  else:
    assert extra is None


def _device_batch(batch: dict) -> dict:
  nd = jax.local_device_count()

  def _expand(arr):
    return jnp.broadcast_to(arr, (nd, *arr.shape))

  return jax.tree.map(_expand, batch)


def _ema_state(model, variables):
  return EMATrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adam(1e-3),
    ema_params=variables["params"],
  )


def _score_setup():
  model = _TinyScore()
  inputs = jnp.ones((2, 4, 4, 3))
  context = jnp.ones((2, 4, 4, 3))
  times = jnp.ones((2,))
  condition = jnp.ones((2, 5))
  variables = model.init(
    {"params": jr.PRNGKey(0)},
    inputs=inputs,
    context=context,
    times=times,
    condition=condition,
    is_training=False,
  )
  state = _ema_state(model, variables)
  batch = {"inputs": inputs, "context": context, "condition": condition}
  return model, state, batch


@pytest.mark.parametrize(
  ("factory_result", "expect_sample_fn", "expect_extra"),
  [
    (
      flow_matching(_TinyScore().apply, FlowMatchingConfig(2), 0.999),
      True,
      False,
    ),
    (
      denoising_diffusion(_TinyScore().apply, EDMParameterization(2), 0.999),
      True,
      False,
    ),
    (
      diffusion_consistency_distillation(
        _TinyScore().apply,
        EDMConsistencyDistillationConfig(2, "edm_style"),
        0.999,
      ),
      True,
      False,
    ),
    (
      wgan(
        _TinyGenerator().apply,
        _TinyCritic().apply,
        WGANConfig(),
        0.999,
      ),
      True,
      False,
    ),
    (
      autoencoder(_TinyVAE(2).apply, AutoencoderConfig(kl_weight=0.1)),
      True,
      True,
    ),
    (
      classifier(
        _TinyClassifier(4).apply,
        ClassifierConfig(weights=jnp.ones((4,))),
      ),
      False,
      False,
    ),
  ],
)
def test_factory_return_contract(
  factory_result, expect_sample_fn, expect_extra
):
  _assert_factory_contract(
    factory_result,
    expect_sample_fn=expect_sample_fn,
    expect_extra=expect_extra,
  )


def test_flow_matching_step_and_eval_finite():
  model, state, batch = _score_setup()
  (step_fn, eval_fn), sample_fn, _ = flow_matching(
    model.apply, FlowMatchingConfig(2), ema_rate=0.999
  )
  nd = jax.local_device_count()
  pbatch = _device_batch(batch)
  pstate = jax_utils.replicate(state)
  pstep = jax.pmap(step_fn, axis_name="batch")
  peval = jax.pmap(eval_fn, axis_name="batch")

  metrics, pstate = pstep(jr.split(jr.PRNGKey(1), nd), pstate, pbatch)
  assert jnp.isfinite(metrics["loss"]).all()

  eval_metrics = peval(jr.split(jr.PRNGKey(2), nd), pstate, pbatch)
  assert jnp.isfinite(eval_metrics["loss"]).all()

  samples = sample_fn(
    jr.PRNGKey(3),
    jax_utils.unreplicate(pstate),
    batch["context"],
    batch["condition"],
  )
  assert jnp.isfinite(samples).all()


def test_denoising_diffusion_step_finite():
  model, state, batch = _score_setup()
  (step_fn, eval_fn), _, _ = denoising_diffusion(
    model.apply, EDMParameterization(2), ema_rate=0.999
  )
  nd = jax.local_device_count()
  pbatch = _device_batch(batch)
  pstate = jax_utils.replicate(state)
  metrics, _ = jax.pmap(step_fn, axis_name="batch")(
    jr.split(jr.PRNGKey(1), nd), pstate, pbatch
  )
  assert jnp.isfinite(metrics["loss"]).all()
  eval_metrics = jax.pmap(eval_fn, axis_name="batch")(
    jr.split(jr.PRNGKey(2), nd), pstate, pbatch
  )
  assert jnp.isfinite(eval_metrics["loss"]).all()


def test_diffusion_consistency_distillation_step_finite():
  model, state, batch = _score_setup()
  (step_fn, eval_fn), _, _ = diffusion_consistency_distillation(
    model.apply,
    EDMConsistencyDistillationConfig(2, "edm_style"),
    ema_rate=0.999,
  )
  nd = jax.local_device_count()
  pbatch = _device_batch(batch)
  pstate = jax_utils.replicate(state)
  metrics, _ = jax.pmap(step_fn, axis_name="batch", in_axes=(0, 0, 0, None))(
    jr.split(jr.PRNGKey(1), nd), pstate, pbatch, 0
  )
  assert jnp.isfinite(metrics["loss"]).all()
  eval_metrics = jax.pmap(eval_fn, axis_name="batch", in_axes=(0, 0, 0, None))(
    jr.split(jr.PRNGKey(2), nd), pstate, pbatch, 0
  )
  assert jnp.isfinite(eval_metrics["loss"]).all()


def test_wgan_step_finite():
  inputs = jnp.ones((2, 4, 4, 3))
  context = jnp.ones((2, 4, 4, 3))
  condition = jnp.ones((2, 5))
  gen = _TinyGenerator()
  critic = _TinyCritic()
  gen_vars = gen.init(
    {"params": jr.PRNGKey(0), "sample": jr.PRNGKey(1)},
    context=context,
    condition=condition,
    is_training=False,
  )
  critic_vars = critic.init(
    {"params": jr.PRNGKey(2)},
    inputs=inputs,
    context=context,
    condition=condition,
    is_training=False,
  )
  generator_state = EMATrainState.create(
    apply_fn=gen.apply,
    params=gen_vars["params"],
    tx=optax.adam(1e-3),
    ema_params=gen_vars["params"],
  )
  critic_state = TrainState.create(
    apply_fn=critic.apply,
    params=critic_vars["params"],
    tx=optax.adam(1e-3),
  )
  (step_fn, eval_fn), _, _ = wgan(
    gen.apply, critic.apply, WGANConfig(), ema_rate=0.999
  )
  batch = {"inputs": inputs, "context": context, "condition": condition}
  nd = jax.local_device_count()
  pbatch = _device_batch(batch)
  state = WGANState(critic=critic_state, generator=generator_state, step=0)
  pstate = jax_utils.replicate(state)
  pstep = jax.pmap(step_fn, axis_name="batch")
  metrics, _ = pstep(jr.split(jr.PRNGKey(3), nd), pstate, pbatch)
  assert jnp.isfinite(metrics["loss_c"]).all()
  assert jnp.isfinite(metrics["loss_g"]).all()


def test_autoencoder_step_finite():
  model = _TinyVAE(channels=2)
  x = jnp.ones((2, 8, 8, 2))
  variables = model.init(
    {"params": jr.PRNGKey(0), "sample": jr.PRNGKey(1)}, x, is_training=False
  )
  state = TrainState.create(
    apply_fn=model.apply, params=variables["params"], tx=optax.adam(1e-3)
  )
  (step_fn, eval_fn), _, reconstruct_fn = autoencoder(
    model.apply, AutoencoderConfig(kl_weight=0.1)
  )
  nd = jax.local_device_count()
  batch = _device_batch({"inputs": x})
  pstate = jax_utils.replicate(state)
  metrics, _ = jax.pmap(step_fn, axis_name="batch")(
    jr.split(jr.PRNGKey(2), nd), pstate, batch
  )
  assert jnp.isfinite(metrics["loss"]).all()
  eval_metrics = jax.pmap(eval_fn, axis_name="batch")(
    jr.split(jr.PRNGKey(3), nd), pstate, batch
  )
  assert jnp.isfinite(eval_metrics["loss"]).all()
  output_hat, _ = jax.pmap(reconstruct_fn, axis_name="batch")(
    jr.split(jr.PRNGKey(4), nd), pstate, batch
  )
  assert jnp.isfinite(output_hat).all()


def test_classifier_step_finite():
  n_classes = 4
  model = _TinyClassifier(n_classes=n_classes)
  x = jnp.ones((2, 8, 8, 2))
  variables = model.init({"params": jr.PRNGKey(0)}, x, is_training=False)
  state = TrainState.create(
    apply_fn=model.apply, params=variables["params"], tx=optax.adam(1e-3)
  )
  (step_fn, eval_fn), _, _ = classifier(
    model.apply, ClassifierConfig(weights=jnp.ones((n_classes,)))
  )
  nd = jax.local_device_count()
  batch = _device_batch(
    {"inputs": x, "label": jnp.zeros((2,), dtype=jnp.int32)}
  )
  pstate = jax_utils.replicate(state)
  metrics, _ = jax.pmap(step_fn, axis_name="batch")(
    jr.split(jr.PRNGKey(1), nd), pstate, batch
  )
  assert jnp.isfinite(metrics["loss"]).all()
  eval_metrics = jax.pmap(eval_fn, axis_name="batch")(
    jr.split(jr.PRNGKey(2), nd), pstate, batch
  )
  assert jnp.isfinite(eval_metrics["loss"]).all()
  assert jnp.isfinite(eval_metrics["accuracy"]).all()


def test_denoising_diffusion_sample_finite():
  model, state, batch = _score_setup()
  _, sample_fn, _ = denoising_diffusion(
    model.apply, EDMParameterization(2), ema_rate=0.999
  )
  samples = sample_fn(
    jr.PRNGKey(0),
    state,
    batch["context"],
    batch["condition"],
  )
  assert samples.shape == batch["context"].shape
  assert jnp.isfinite(samples).all()


def test_wgan_eval_and_sample_finite():
  inputs = jnp.ones((2, 4, 4, 3))
  context = jnp.ones((2, 4, 4, 3))
  condition = jnp.ones((2, 5))
  gen = _TinyGenerator()
  critic = _TinyCritic()
  gen_vars = gen.init(
    {"params": jr.PRNGKey(0), "sample": jr.PRNGKey(1)},
    context=context,
    condition=condition,
    is_training=False,
  )
  critic_vars = critic.init(
    {"params": jr.PRNGKey(2)},
    inputs=inputs,
    context=context,
    condition=condition,
    is_training=False,
  )
  generator_state = EMATrainState.create(
    apply_fn=gen.apply,
    params=gen_vars["params"],
    tx=optax.adam(1e-3),
    ema_params=gen_vars["params"],
  )
  critic_state = TrainState.create(
    apply_fn=critic.apply,
    params=critic_vars["params"],
    tx=optax.adam(1e-3),
  )
  (_, eval_fn), sample_fn, _ = wgan(
    gen.apply, critic.apply, WGANConfig(), ema_rate=0.999
  )
  batch = {"inputs": inputs, "context": context, "condition": condition}
  nd = jax.local_device_count()
  pbatch = _device_batch(batch)
  state = WGANState(critic=critic_state, generator=generator_state, step=0)
  pstate = jax_utils.replicate(state)

  eval_metrics = jax.pmap(eval_fn, axis_name="batch")(
    jr.split(jr.PRNGKey(0), nd), pstate, pbatch
  )
  assert jnp.isfinite(eval_metrics["loss_c"]).all()
  assert jnp.isfinite(eval_metrics["loss_g"]).all()
  assert jnp.isfinite(eval_metrics["synthetic_crit"]).all()
  assert jnp.isfinite(eval_metrics["inputs_critic"]).all()

  samples = sample_fn(
    jr.PRNGKey(1), jax_utils.unreplicate(pstate), context, condition
  )
  assert samples.shape == context.shape
  assert jnp.isfinite(samples).all()


def test_wgan_step_counter_increments():
  inputs = jnp.ones((2, 4, 4, 3))
  context = jnp.ones((2, 4, 4, 3))
  condition = jnp.ones((2, 5))
  gen = _TinyGenerator()
  critic = _TinyCritic()
  gen_vars = gen.init(
    {"params": jr.PRNGKey(0), "sample": jr.PRNGKey(1)},
    context=context,
    condition=condition,
    is_training=False,
  )
  critic_vars = critic.init(
    {"params": jr.PRNGKey(2)},
    inputs=inputs,
    context=context,
    condition=condition,
    is_training=False,
  )
  generator_state = EMATrainState.create(
    apply_fn=gen.apply,
    params=gen_vars["params"],
    tx=optax.adam(1e-3),
    ema_params=gen_vars["params"],
  )
  critic_state = TrainState.create(
    apply_fn=critic.apply,
    params=critic_vars["params"],
    tx=optax.adam(1e-3),
  )
  (step_fn, _), _, _ = wgan(
    gen.apply, critic.apply, WGANConfig(), ema_rate=0.999
  )
  batch = {"inputs": inputs, "context": context, "condition": condition}
  nd = jax.local_device_count()
  pbatch = _device_batch(batch)
  state = WGANState(critic=critic_state, generator=generator_state, step=0)
  pstate = jax_utils.replicate(state)
  _, pstate2 = jax.pmap(step_fn, axis_name="batch")(
    jr.split(jr.PRNGKey(0), nd), pstate, pbatch
  )
  assert int(jax_utils.unreplicate(pstate2).step) == 1


def test_cd_sample_finite():
  model, state, batch = _score_setup()
  _, sample_fn, _ = diffusion_consistency_distillation(
    model.apply,
    EDMConsistencyDistillationConfig(2, "edm_style"),
    ema_rate=0.999,
  )
  samples = sample_fn(
    jr.PRNGKey(0),
    state,
    batch["context"],
    batch["condition"],
  )
  assert samples.shape == batch["context"].shape
  assert jnp.isfinite(samples).all()
