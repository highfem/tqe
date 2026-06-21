"""Classifier; returns ``((step_fn, eval_fn), None, None)``."""

import dataclasses
from collections.abc import Callable

import jax
import optax
from jax import numpy as jnp

from tqe._types import ObjectiveFns, TrainFns


@dataclasses.dataclass
class ClassifierConfig:
  """Per-class loss weights for weighted cross-entropy.

  Attributes:
    weights: Shape ``(num_classes,)``.
  """

  weights: jnp.ndarray


def classifier(model_fn: Callable, config: ClassifierConfig) -> ObjectiveFns:
  """Build classifier training functions.

  Args:
    model_fn: Flax ``apply``; returns ``(logits, _)``.
    config: Classifier configuration.

  Returns:
    ``((step_fn, eval_fn), None, None)``. ``step_fn`` / ``eval_fn`` use batch
    keys ``inputs`` and ``label``. Loss and gradients are pmap-reduced.
  """
  weights = config.weights

  def _apply_classifier(params, rng_key, inputs, is_training):
    return model_fn(
      variables={"params": params},
      rngs={"dropout": rng_key},
      inputs=inputs,
      is_training=is_training,
    )

  def weighted_cross_entropy(logits, labels):
    labels = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels)
    batch_weights = jnp.sum(labels * weights, axis=-1)
    loss = (loss * batch_weights).sum() / batch_weights.sum()
    return loss

  def loss_fn(params, rng_key, batch, is_training):
    logits, _ = _apply_classifier(params, rng_key, batch["inputs"], is_training)
    return weighted_cross_entropy(logits, batch["label"])

  @jax.jit
  def step_fn(rng_key, state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, rng_key, batch, True)
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    return {"loss": loss}, new_state

  @jax.jit
  def eval_fn(rng_key, state, batch):
    logits, _ = _apply_classifier(state.params, rng_key, batch["inputs"], False)
    labels = batch["label"]
    loss = weighted_cross_entropy(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = jax.lax.pmean(
      {"loss": loss, "accuracy": accuracy}, axis_name="batch"
    )
    return metrics

  return ObjectiveFns(TrainFns(step_fn, eval_fn), None, None)
