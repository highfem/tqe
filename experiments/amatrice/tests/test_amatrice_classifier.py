import jax
import optax
from flax import jax_utils
from flax import linen as fnn
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import random as jr

from tqe.classifier import ClassifierConfig, classifier


class _TinyClassifier(fnn.Module):
  n_classes: int

  @fnn.compact
  def __call__(self, inputs, is_training):
    h = inputs.reshape((inputs.shape[0], -1))
    logits = fnn.Dense(self.n_classes)(h)
    return logits, None


def test_classifier_objective_runs_step_and_eval():
  n_classes = 4
  model = _TinyClassifier(n_classes=n_classes)
  x = jnp.ones((2, 8, 8, 2))
  variables = model.init({"params": jr.PRNGKey(0)}, x, is_training=False)
  state = TrainState.create(
    apply_fn=model.apply, params=variables["params"], tx=optax.adam(1e-3)
  )
  weights = jnp.ones((n_classes,))
  (step_fn, eval_fn), _, _ = classifier(
    model.apply, ClassifierConfig(weights=weights)
  )

  nd = jax.local_device_count()
  pstate = jax_utils.replicate(state)
  batch = {
    "inputs": jnp.ones((nd, 2, 8, 8, 2)),
    "label": jnp.zeros((nd, 2), dtype=jnp.int32),
  }
  pstep = jax.pmap(step_fn, axis_name="batch")
  peval = jax.pmap(eval_fn, axis_name="batch")

  metrics, pstate = pstep(jr.split(jr.PRNGKey(1), nd), pstate, batch)
  assert jnp.isfinite(metrics["loss"]).all()
  eval_metrics = peval(jr.split(jr.PRNGKey(2), nd), pstate, batch)
  assert jnp.isfinite(eval_metrics["loss"]).all()
  assert jnp.isfinite(eval_metrics["accuracy"]).all()
