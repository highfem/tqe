import jax
import optax
from flax import jax_utils
from flax import linen as fnn
from jax import numpy as jnp
from jax import random as jr
from train_state import EMATrainState

from tqe.flow_matching import FlowMatchingConfig, flow_matching


class _TinyScore(fnn.Module):
  @fnn.compact
  def __call__(self, inputs, context, times, condition, is_training):
    h = jnp.concatenate([inputs, context], axis=-1)
    return fnn.Dense(inputs.shape[-1])(h)


def _make_state(model, variables):
  return EMATrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adam(1e-3),
    ema_params=variables["params"],
  )


def test_flow_matching_step_eval_and_sample_finite():
  model = _TinyScore()
  x = jnp.ones((2, 4, 4, 3))
  ctx = jnp.ones((2, 4, 4, 3))
  times = jnp.ones((2,))
  cond = jnp.ones((2, 5))
  variables = model.init(
    {"params": jr.PRNGKey(0)},
    inputs=x,
    context=ctx,
    times=times,
    condition=cond,
    is_training=False,
  )
  state = _make_state(model, variables)
  (step_fn, eval_fn), sample_fn, _ = flow_matching(
    model.apply, FlowMatchingConfig(2), ema_rate=0.999
  )

  nd = jax.local_device_count()
  pstate = jax_utils.replicate(state)
  batch = {"inputs": x, "context": ctx, "condition": cond}
  batch = jax.tree.map(
    lambda arr: jnp.broadcast_to(arr, (nd, *arr.shape)), batch
  )

  pstep = jax.pmap(step_fn, axis_name="batch")
  peval = jax.pmap(eval_fn, axis_name="batch")
  metrics, pstate = pstep(jr.split(jr.PRNGKey(1), nd), pstate, batch)
  assert jnp.isfinite(metrics["loss"]).all()

  eval_metrics = peval(jr.split(jr.PRNGKey(2), nd), pstate, batch)
  assert jnp.isfinite(eval_metrics["loss"]).all()

  samples = sample_fn(jr.PRNGKey(3), jax_utils.unreplicate(pstate), ctx, cond)
  assert jnp.isfinite(samples).all()
