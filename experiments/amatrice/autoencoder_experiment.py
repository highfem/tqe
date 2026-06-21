import os

import jax
import train_state
import wandb
from absl import logging
from checkpoint import get_checkpoint_manager, get_latest_train_state
from data import get_train_iters
from flax import jax_utils
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from utils import callback_log, callback_summarize, metrics_to_summary

from tqe.autoencoder import AutoencoderConfig, autoencoder
from tqe.nn import vae
from tqe.signal import processing_fns


def _new_train_state(rng_key, config):
  logging.info("initializing autoencoder")
  params_key, sample_key = jr.split(rng_key)
  model = vae.make_model(config.model.nn)
  batch = jnp.ones(
    (jax.device_count(), *config.data.image_size, config.data.n_out_channels)
  )
  variables = model.init(
    {"params": params_key, "sample": sample_key},
    batch,
    is_training=False,
  )
  n_params = jax.tree.map(lambda x: x.size, variables["params"])
  n_params = jnp.sum(jnp.array(jax.tree.leaves(n_params)))
  logging.info(f"total number of parameters of model: {n_params}")
  state = train_state.new_train_state(variables, model, config)
  return state, model


def train(rng_key, FLAGS, callbacks, run_name):
  config = FLAGS.config
  workdir = FLAGS.workdir
  use_wandb = FLAGS.usewand
  iter_key, init_key, rng_key = jr.split(jr.PRNGKey(rng_key), 3)
  repr_fn, inv_repr_fn = processing_fns(
    variable_names=config.model_type.replace("_encoder", ""),
    repres=config.representation,
    max_len=config.representation.max_len,
  )
  train_iter, val_iter = get_train_iters(iter_key, config, workdir, repr_fn)

  state, model = _new_train_state(init_key, config)
  (step_fn, eval_fn), _, reconstruct_fn = autoencoder(
    model.apply, AutoencoderConfig(config.training.params.kl_weight)
  )

  mngr, ckpt_save_fn, _ = get_checkpoint_manager(
    os.path.join(workdir, "checkpoints", run_name),
    config,
    config.model.to_dict(),
  )
  state, cmetrics, cstep = get_latest_train_state(mngr, state)

  pstep_fn = jax.pmap(step_fn, axis_name="batch")
  peval_fn = jax.pmap(eval_fn, axis_name="batch")
  preconstruct_fn = jax.pmap(reconstruct_fn, axis_name="batch")
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
    train_key, val_key, sample_key = jr.split(jr.fold_in(step_key, step), 3)
    metrics, pstate = pstep_fn(
      jr.split(train_key, jax.device_count()), pstate, pbatch
    )
    train_metrics.append(metrics)
    is_first_step = step == 1
    is_last_step = step == config.training.n_steps
    is_first_or_last_step = is_first_step or is_last_step

    if step % config.training.n_eval_frequency == 0 or is_first_or_last_step:
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
          logging.info(f"found new best in step {step}: {best_val_loss}")
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

    if step % config.training.n_sampling_frequency == 0 or is_last_step:
      summaries = {}
      for sample_idx, sbatch in zip(
        range(config.training.n_sampling_batches), val_iter
      ):
        specs_hat, specs_latent_hat = preconstruct_fn(
          jr.split(jr.fold_in(sample_key, sample_idx), jax.device_count()),
          pstate,
          sbatch,
        )
        specs_latent_hat = specs_latent_hat.reshape(
          (-1, *specs_latent_hat.shape[2:])
        )
        specs = sbatch["inputs"].reshape((-1, *sbatch["inputs"].shape[2:]))
        specs_hat = specs_hat.reshape((-1, *sbatch["inputs"].shape[2:]))
        signals = sbatch["signal"].reshape((-1, *sbatch["signal"].shape[2:]))
        signals_recovered = inv_repr_fn(specs)
        signals_hat = inv_repr_fn(specs_hat)
        summaries = callback_summarize(
          step, callbacks, signals, signals_hat, summaries
        )
      if jax.process_index() == 0:
        logging.info("plotting reconstructions")
        plot_signals(
          step,
          signals[:5],
          signals_recovered[:5],
          signals_hat[:5],
          specs[:5],
          specs_hat[:5],
          specs_latent_hat[:5],
          use_wandb,
        )
        callback_log(use_wandb, step, callbacks, summaries)
  logging.info("finished training")


def plot_signals(
  step,
  target_signals,
  target_signals_recovered,
  target_signals_hat,
  target_specs,
  target_specs_hat,
  target_specs_latent_hat,
  use_wandb,
):
  fig, axes = plt.subplots(figsize=(20, 20), ncols=5, nrows=5)
  for i in range(min(5, target_signals.shape[0])):
    axes[i, 0].set_title("True signal")
    axes[i, 0].plot(target_signals[i], alpha=0.5)
    axes[i, 1].set_title("Reconstructed signal")
    axes[i, 1].plot(target_signals_hat[i], alpha=0.5)
    axes[i, 2].set_title("True spectrogram")
    axes[i, 2].imshow(target_specs[i, :, :, 0], alpha=0.5)
    axes[i, 3].set_title("Trained spectrogram")
    axes[i, 3].imshow(target_specs_hat[i, :, :, 0], alpha=0.5)
    axes[i, 4].set_title("Latent spectrogram")
    axes[i, 4].imshow(target_specs_latent_hat[i, :, :, 0], alpha=0.5)
  plt.tight_layout()
  plt.close()
  if use_wandb:
    wandb.log(
      {"Original data and reconstructions": wandb.Image(fig)}, step=step
    )
