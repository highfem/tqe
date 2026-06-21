import os

import jax
import latent_common
import train_state
import wandb
from absl import logging
from checkpoint import get_checkpoint_manager, get_latest_train_state
from data import get_train_iters
from flax import jax_utils
from jax import numpy as jnp
from jax import random as jr
from utils import (
  callback_log,
  callback_summarize,
  metrics_to_summary,
  plot_signals,
)

from tqe.denoising_diffusion import EDMParameterization, denoising_diffusion
from tqe.flow_matching import FlowMatchingConfig, flow_matching
from tqe.nn import dit, unet
from tqe.signal import processing_fns


def get_model_and_matching_fns(config):
  if config.model.nn.name == "dit":
    logging.info("using dit")
    score_model = dit.make_model(config.model.nn.dit_score_net)
  else:
    logging.info("using unet")
    score_model = unet.make_model(config.model.nn.unet_score_net)
  ema_rate = config.training.ema_rate
  if config.model.name == "flow_matching":
    logging.info("using flow matcher")
    matching_fns = flow_matching(
      score_model.apply,
      FlowMatchingConfig(config.model.sampler.n_steps),
      ema_rate,
    )
  elif config.model.name == "edm":
    logging.info("using edm")
    matching_fns = denoising_diffusion(
      score_model.apply,
      EDMParameterization(config.model.sampler.n_steps),
      ema_rate,
    )
  else:
    raise ValueError(f"unknown model {config.model.name!r}")
  return score_model, matching_fns


def _new_train_state(
  rng_key,
  config,
  workdir,
  autoencoder_checkpoints,
  autoencoder_checkpoints_steps,
):
  logging.info("initializing latent diffusion")
  model, _ = get_model_and_matching_fns(config)
  (target_enc_fn, context_enc_fn), _ = latent_common.get_encoders_and_decoders(
    workdir, autoencoder_checkpoints, autoencoder_checkpoints_steps
  )

  params_key, sample_key = jr.split(rng_key)
  batch = jnp.ones(
    (jax.device_count(), *config.data.image_size, config.data.n_out_channels)
  )
  times = jnp.ones(jax.device_count())
  condition = jnp.ones((jax.device_count(), config.data.condition_length))

  if target_enc_fn is not None:
    targets = target_enc_fn({"sample": jr.PRNGKey(0)}, batch)
    context = context_enc_fn({"sample": jr.PRNGKey(0)}, batch)
    logging.info(f"pbatch shape target lowdim: {targets.shape}")
    logging.info(f"pbatch shape context lowdim: {context.shape}")
  else:
    targets = context = batch

  variables = model.init(
    {"params": params_key, "sample": sample_key},
    inputs=targets,
    context=context,
    times=times,
    condition=condition,
    is_training=False,
  )
  n_params = jax.tree.map(lambda x: x.size, variables["params"])
  n_params = jnp.sum(jnp.array(jax.tree.leaves(n_params)))
  logging.info(f"total number of parameters of model: {n_params}")
  state = train_state.new_ema_train_state(variables, model, config)
  return state, model


def get_sample_fn(
  sampling_fn, target_encoder_fn, context_encoder_fn, target_decoder_fn
):
  @jax.jit
  def sample_fn(rng_key, state, batch):
    sampling_key, context_key, target_key = jr.split(rng_key, 3)
    context = batch["context"]
    if context_encoder_fn is not None:
      context = context_encoder_fn({"sample": context_key}, batch["context"])

    spec_latent_hat = sampling_fn(
      sampling_key,
      state=state,
      context=context,
      condition=batch["meta"],
    )

    spec_latent = spec_hat = spec_from_enc = None
    if target_encoder_fn is not None and "target" in batch:
      spec_latent = target_encoder_fn({"sample": target_key}, batch["target"])
      if target_decoder_fn is not None:
        spec_from_enc = target_decoder_fn(None, spec_latent)
    if target_decoder_fn is not None:
      spec_hat = target_decoder_fn(None, spec_latent_hat)

    if target_encoder_fn is not None:
      return (spec_latent_hat, spec_hat), (spec_latent, spec_from_enc)
    return (None, spec_latent_hat), (None, None)

  return sample_fn


def train(rng_key, FLAGS, callbacks, run_name):
  config = FLAGS.config
  workdir = FLAGS.workdir
  use_wandb = FLAGS.usewand
  iter_key, init_key, rng_key = jr.split(jr.PRNGKey(rng_key), 3)

  _, matching_fns = get_model_and_matching_fns(config)
  (step_eval, sampling_fn, _) = matching_fns
  step_fn, eval_fn = step_eval
  (target_encoder_fn, context_encoder_fn), (target_decoder_fn, _) = (
    latent_common.get_encoders_and_decoders(
      workdir,
      FLAGS.autoencoder_checkpoints,
      FLAGS.autoencoder_checkpoints_steps,
    )
  )
  repr_fn, inv_repr_fn = processing_fns(
    variable_names=["target", "context"],
    repres=config.representation,
    max_len=config.representation.max_len,
  )
  train_iter, val_iter = get_train_iters(iter_key, config, workdir, repr_fn)
  state, _ = _new_train_state(
    init_key,
    config,
    workdir,
    FLAGS.autoencoder_checkpoints,
    FLAGS.autoencoder_checkpoints_steps,
  )
  mngr, ckpt_save_fn, _ = get_checkpoint_manager(
    os.path.join(workdir, "checkpoints", run_name),
    config,
    config.model.to_dict(),
  )
  state, cmetrics, cstep = get_latest_train_state(mngr, state)

  pstep_fn = jax.pmap(
    latent_common.wrap_step(step_fn, target_encoder_fn, context_encoder_fn),
    axis_name="batch",
  )
  peval_fn = jax.pmap(
    latent_common.wrap_eval(eval_fn, target_encoder_fn, context_encoder_fn),
    axis_name="batch",
  )
  p_sample_fn = jax.pmap(
    get_sample_fn(
      sampling_fn, target_encoder_fn, context_encoder_fn, target_decoder_fn
    ),
    axis_name="batch",
  )
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
    if step == 1 and jax.process_index() == 0:
      logging.info(f"pbatch shape: {pbatch['target'].shape}")
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
      specs_from_enc = None
      for sample_idx, sbatch in zip(
        range(config.training.n_sampling_batches), val_iter
      ):
        (
          (specs_latents_hat, specs_hat),
          (
            specs_latent_train,
            specs_from_enc,
          ),
        ) = p_sample_fn(
          jr.split(jr.fold_in(sample_key, sample_idx), jax.device_count()),
          pstate,
          sbatch,
        )
        if specs_latent_train is not None:
          specs_latents_hat = specs_latents_hat.reshape(
            (-1, *specs_latents_hat.shape[2:])
          )
          specs_latent_train = specs_latent_train.reshape(
            (-1, *specs_latent_train.shape[2:])
          )
          specs_from_enc = specs_from_enc.reshape(
            (-1, *specs_from_enc.shape[2:])
          )
        specs_hat = specs_hat.reshape((-1, *specs_hat.shape[2:]))
        specs = sbatch["target"].reshape((-1, *sbatch["target"].shape[2:]))
        signals = sbatch["signal"].reshape((-1, *sbatch["signal"].shape[2:]))
        signals_hat = inv_repr_fn(specs_hat)
        summaries = callback_summarize(
          step, callbacks, signals, signals_hat, summaries
        )
      if jax.process_index() == 0:
        signals_from_enc = None
        if specs_from_enc is not None:
          signals_from_enc = inv_repr_fn(specs_from_enc)
        plot_signals(
          step,
          signals[:5],
          signals_from_enc[:5] if signals_from_enc is not None else None,
          signals_hat[:5],
          specs[:5],
          specs_from_enc[:5] if specs_from_enc is not None else None,
          specs_hat[:5],
          specs_latent_train[:5] if specs_latent_train is not None else None,
          specs_latents_hat[:5] if specs_latents_hat is not None else None,
          use_wandb,
        )
        callback_log(use_wandb, step, callbacks, summaries)
  logging.info("finished training")
