import os

import jax
import train_state
import wandb
from absl import logging
from checkpoint import get_checkpoint_manager, get_latest_train_state
from data import data_loaders, get_train_iters
from flax import jax_utils
from jax import numpy as jnp
from jax import random as jr
from utils import metrics_to_summary

from tqe.classifier import ClassifierConfig, classifier
from tqe.nn import inception
from tqe.signal import processing_fns


def _labels_from_meta(meta, distance_bins, magnitude_bins):
  distance = jnp.sqrt(
    jnp.square(meta[:, 6] - meta[:, 2])
    + jnp.square(meta[:, 7] - meta[:, 3])
    + jnp.square(meta[:, 5] - meta[:, 4] * 1e-3)
  )
  magnitude = meta[:, 0]
  distance_bins = jnp.array(distance_bins)
  magnitude_bins = jnp.array(magnitude_bins)
  labels = (jnp.digitize(distance, distance_bins) - 1) * (
    len(magnitude_bins) - 1
  )
  labels += jnp.digitize(magnitude, magnitude_bins) - 1
  return labels


def get_model_and_matching_fns(config, class_weights):
  if config.model.nn.name == "inception_resnet":
    logging.info("using inception_resnet")
    model = inception.make_model(config.model.nn)
  else:
    raise ValueError(f"unknown nn {config.model.nn.name!r}")
  matching_fns = classifier(
    model.apply, ClassifierConfig(weights=class_weights)
  )
  return model, matching_fns


def _new_train_state(rng_key, config):
  logging.info("initializing classifier")
  model, _ = get_model_and_matching_fns(config, None)
  params_key, _ = jr.split(rng_key)
  batch = jnp.ones(
    (
      jax.device_count() * 2,
      *config.data.image_size,
      config.data.n_out_channels,
    )
  )
  variables = model.init(
    {"params": params_key},
    batch,
    is_training=False,
  )
  n_params = jax.tree.map(lambda x: x.size, variables["params"])
  n_params = jnp.sum(jnp.array(jax.tree.leaves(n_params)))
  logging.info(f"total number of parameters of model: {n_params}")
  state = train_state.new_train_state(variables, model, config)
  return state, model


def get_class_weights(rng_key, config, workdir):
  itrs, _ = data_loaders(
    rng_key=rng_key,
    config=config.training,
    dataset=config.data.dataset,
    split=["train", "test"],
    representation_fn=lambda x: x,
    outpath=os.path.join(workdir, "data"),
    repeat=1,
  )
  metas = []
  for _, itr in itrs.items():
    for batch in itr:
      metas.append(batch["meta"])
  metas = jnp.concatenate(metas, axis=0)
  labels = _labels_from_meta(
    metas, config.data.distance_bins, config.data.magnitude_bins
  )
  n_classes = (len(config.data.magnitude_bins) - 1) * (
    len(config.data.distance_bins) - 1
  )
  weights = jnp.array(
    [jnp.reciprocal(jnp.sum(labels == c)) for c in range(n_classes)]
  )
  if n_classes != len(jnp.unique(labels)):
    logging.error(
      f"n_classes != unique labels: {n_classes} vs {len(jnp.unique(labels))}"
    )
  return weights, n_classes


def get_step_or_eval_fn(step_or_eval_fn, config):
  @jax.jit
  def fn(rng_key, state, batch):
    batch = dict(batch)
    batch["label"] = _labels_from_meta(
      batch["meta"], config.data.distance_bins, config.data.magnitude_bins
    )
    return step_or_eval_fn(rng_key, state, batch)

  return fn


def train(rng_key, FLAGS, callbacks, run_name):
  del callbacks
  config = FLAGS.config
  workdir = FLAGS.workdir
  use_wandb = FLAGS.usewand
  class_key, rng_key = jr.split(jr.PRNGKey(rng_key))
  class_weights, _ = get_class_weights(class_key, config, workdir)
  _, matching_fns = get_model_and_matching_fns(config, class_weights)
  (step_fn, eval_fn), _, _ = matching_fns

  repr_fn, _ = processing_fns(
    variable_names="target",
    repres=config.representation,
    max_len=config.representation.max_len,
  )
  iter_key, rng_key = jr.split(rng_key)
  train_iter, val_iter = get_train_iters(iter_key, config, workdir, repr_fn)

  init_key, rng_key = jr.split(rng_key)
  state, _ = _new_train_state(init_key, config)
  mngr, ckpt_save_fn, *_ = get_checkpoint_manager(
    os.path.join(workdir, "checkpoints", run_name),
    config,
    config.model.to_dict(),
  )
  state, cmetrics, cstep = get_latest_train_state(mngr, state)

  pstep_fn = jax.pmap(get_step_or_eval_fn(step_fn, config), axis_name="batch")
  peval_fn = jax.pmap(get_step_or_eval_fn(eval_fn, config), axis_name="batch")
  pstate = jax_utils.replicate(state)

  best_val_loss = cmetrics["val/best"]
  if jax.process_index() == 0:
    logging.info("training model")
    logging.info(f"starting/resuming training at step: {cstep}")
  step_key, rng_key = jr.split(rng_key)
  train_metrics = []
  for step, pbatch in zip(
    range(cstep + 1, config.training.n_steps + 1), train_iter
  ):
    train_key, val_key, _ = jr.split(jr.fold_in(step_key, step), 3)
    metrics, pstate = pstep_fn(
      jr.split(train_key, jax.device_count()), pstate, pbatch
    )
    train_metrics.append(metrics)
    is_first_step = step == 1
    is_last_step = step == config.training.n_steps
    if (
      step % config.training.n_eval_frequency == 0
      or is_first_step
      or is_last_step
    ):
      val_metrics = []
      for val_idx, vbatch in zip(
        range(config.training.n_eval_batches), val_iter
      ):
        metrics = peval_fn(
          jr.split(jr.fold_in(val_key, val_idx), jax.device_count()),
          pstate,
          vbatch,
        )
        val_metrics.append(metrics)
      summary = metrics_to_summary(train_metrics, val_metrics)
      train_metrics = []
      if jax.process_index() == 0:
        logging.info(
          f"step {step} train/val loss: "
          f"{summary['train/loss']}/{summary['val/loss']}"
        )
      if (
        step % config.training.n_checkpointing_frequency == 0
        and jax.process_index() == 0
      ):
        if summary["val/loss"] < best_val_loss:
          best_val_loss = summary["val/loss"]
          ckpt_save_fn(
            step,
            jax_utils.unreplicate(pstate),
            summary | {"val/best": best_val_loss},
            best=True,
          )
        ckpt_save_fn(
          step,
          jax_utils.unreplicate(pstate),
          summary | {"val/best": best_val_loss},
        )
      if use_wandb and jax.process_index() == 0:
        wandb.log(summary, step=step)
  logging.info("finished training")
