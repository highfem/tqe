"""Flax train state, optimizers, and EMA parameter tracking."""

from typing import Any

import optax
from flax import core, struct
from flax.training import train_state
from flax.training.train_state import TrainState
from jax import tree_util


class EMATrainState(train_state.TrainState):
  """``TrainState`` extended with exponential-moving-average model parameters."""

  ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


def get_optimizer(config):
  if config.optimizer.params.do_warmup and config.optimizer.params.do_decay:
    lr = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=config.optimizer.params.learning_rate,
      warmup_steps=config.optimizer.params.warmup_steps,
      decay_steps=config.optimizer.params.decay_steps,
      end_value=config.optimizer.params.end_learning_rate,
    )
  elif config.optimizer.params.do_warmup:
    lr = optax.linear_schedule(
      init_value=0.0,
      end_value=config.optimizer.params.learning_rate,
      transition_steps=config.optimizer.params.warmup_steps,
    )
  elif config.optimizer.params.do_decay:
    lr = optax.cosine_decay_schedule(
      init_value=config.optimizer.params.learning_rate,
      decay_steps=config.optimizer.params.decay_steps,
      alpha=config.optimizer.params.end_learning_rate
      / config.optimizer.params.learning_rate,
    )
  else:
    lr = config.optimizer.params.learning_rate

  if config.optimizer.name == "adamw":
    tx = optax.adamw(
      lr,
      b1=config.optimizer.params.b1,
      b2=config.optimizer.params.b2,
      weight_decay=config.optimizer.params.weight_decay,
    )
  elif config.optimizer.name == "adam":
    tx = optax.adam(
      lr, b1=config.optimizer.params.b1, b2=config.optimizer.params.b2
    )
  elif config.optimizer.name == "sgd":
    tx = optax.sgd(
      lr,
      momentum=config.optimizer.params.b1,
    )
  else:
    raise ValueError(f"unknown optimizer {config.optimizer.name!r}")

  if config.optimizer.params.do_gradient_clipping:
    tx = optax.chain(
      optax.clip_by_global_norm(config.optimizer.params.gradient_clipping),
      tx,
    )
  return tx


def new_train_state(variables, model, config):
  ts = TrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=get_optimizer(config),
  )
  return ts


def new_ema_train_state(variables, model, config):
  ts = EMATrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    ema_params=variables["params"].copy(),
    tx=get_optimizer(config),
  )
  return ts


def restore_optimizer_state(opt_state, restored):
  return tree_util.tree_unflatten(
    tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
  )


def restore_train_state(mngr, ts, which):
  step = mngr.latest_step() if which == "latest" else mngr.best_step()
  restored_dict = mngr.restore(step)
  return _restore_state(ts, restored_dict)


def _restore_state(ts, restored_dict):
  restored_optimizer = restore_optimizer_state(
    ts.opt_state, restored_dict["state"]["opt_state"]
  )
  if "ema_params" not in restored_dict["state"]:
    ts = ts.replace(
      params=restored_dict["state"]["params"],
      step=restored_dict["state"]["step"],
      opt_state=restored_optimizer,
    )
  else:
    ts = ts.replace(
      params=restored_dict["state"]["params"],
      ema_params=restored_dict["state"]["ema_params"].copy(),
      step=restored_dict["state"]["step"],
      opt_state=restored_optimizer,
    )
  return ts, restored_dict["metrics"]
