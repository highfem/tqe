import jax
import optax
from flax import jax_utils
from flax import linen as fnn
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import random as jr

from tqe.autoencoder import AutoencoderConfig, autoencoder


class _TinyVAE(fnn.Module):
  channels: int

  @fnn.compact
  def __call__(self, inputs, is_training):
    h = fnn.Conv(8, (3, 3), padding="SAME")(inputs)
    shift = fnn.Conv(self.channels, (3, 3), padding="SAME")(h)
    log_scale = fnn.Conv(self.channels, (3, 3), padding="SAME")(h)
    z = shift + jnp.exp(log_scale) * jr.normal(
      self.make_rng("sample"), log_scale.shape
    )
    output_hat = fnn.Conv(self.channels, (3, 3), padding="SAME")(z)
    return output_hat, (z, shift, log_scale)


def _make_state():
  model = _TinyVAE(channels=2)
  x = jnp.ones((1, 8, 8, 2))
  variables = model.init(
    {"params": jr.PRNGKey(0), "sample": jr.PRNGKey(1)}, x, is_training=False
  )
  state = TrainState.create(
    apply_fn=model.apply, params=variables["params"], tx=optax.adam(1e-3)
  )
  return model, state


def test_autoencoder_objective_runs_step_eval_reconstruct():
  model, state = _make_state()
  (step_fn, eval_fn), _, reconstruct_fn = autoencoder(
    model.apply, AutoencoderConfig(kl_weight=0.1)
  )

  nd = jax.local_device_count()
  pstate = jax_utils.replicate(state)
  batch = {"inputs": jnp.ones((nd, 4, 8, 8, 2))}

  pstep = jax.pmap(step_fn, axis_name="batch")
  peval = jax.pmap(eval_fn, axis_name="batch")
  precon = jax.pmap(reconstruct_fn, axis_name="batch")

  metrics, pstate = pstep(jr.split(jr.PRNGKey(2), nd), pstate, batch)
  assert jnp.isfinite(metrics["loss"]).all()

  eval_metrics = peval(jr.split(jr.PRNGKey(3), nd), pstate, batch)
  assert jnp.isfinite(eval_metrics["loss"]).all()

  output_hat, shift = precon(jr.split(jr.PRNGKey(4), nd), pstate, batch)
  assert output_hat.shape == batch["inputs"].shape
  assert shift.shape == batch["inputs"].shape
