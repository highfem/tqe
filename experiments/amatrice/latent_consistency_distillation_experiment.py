import os

import jax
import latent_common
import train_state
import wandb
from absl import logging
from checkpoint import (
  get_checkpoint_manager,
  get_latest_train_state,
  load_dit_with_ema_weights,
)
from data import get_train_iters
from flax import jax_utils
from jax import numpy as jnp
from jax import random as jr
from latent_diffusion_experiment import get_sample_fn
from utils import (
  callback_log,
  callback_summarize,
  metrics_to_summary,
  plot_spectrograms,
)

from tqe.diffusion_consistency_distillation import (
  EDMConsistencyDistillationConfig,
  diffusion_consistency_distillation,
)
from tqe.nn import dit
from tqe.signal import processing_fns


def get_model_and_matching_fns(config):
  if config.model.nn.name == "dit":
    logging.info("using dit")
    score_model = dit.make_model(config.model.nn.dit_score_net)
  else:
    raise ValueError(f"unknown nn {config.model.nn.name!r}")
  if config.model.name == "flow_matching":
    raise ValueError(
      "flow consistency distillation is not available in tqe; use edm"
    )
  logging.info(
    f"using edm consistency distillator with type {config.model.distillation.type}"
  )
  matching_fns = diffusion_consistency_distillation(
    score_model.apply,
    EDMConsistencyDistillationConfig(
      config.model.sampler.n_steps, config.model.distillation.type
    ),
    config.training.ema_rate,
  )
  return score_model, matching_fns


def get_teacher_model(FLAGS, config):
  pth = os.path.join(FLAGS.workdir, "checkpoints", FLAGS.ddm_checkpoint)
  return load_dit_with_ema_weights(pth, FLAGS.ddm_checkpoint_step)


def _new_train_state(rng_key, FLAGS, config, teacher_params):
  logging.info("initializing latent consistency distillation")
  model, _ = get_model_and_matching_fns(config)
  teacher_params = {"params": teacher_params}
  n_params = jax.tree.map(lambda x: x.size, teacher_params["params"])
  n_params = jnp.sum(jnp.array(jax.tree.leaves(n_params)))
  logging.info(f"total number of parameters of model: {n_params}")
  state = train_state.new_ema_train_state(teacher_params, model, config)
  return state, model


def train(rng_key, FLAGS, callbacks, run_name):
  iter_key, init_key, rng_key = jr.split(jr.PRNGKey(rng_key), 3)

  _, matching_fns = get_model_and_matching_fns(FLAGS.config)
  (step_fn, eval_fn), sampling_fn, _ = matching_fns
  _, teacher_params = get_teacher_model(FLAGS, FLAGS.config)
  (target_encoder_fn, context_encoder_fn), (target_decoder_fn, _) = (
    latent_common.get_encoders_and_decoders(
      FLAGS.workdir,
      FLAGS.autoencoder_checkpoints,
      FLAGS.autoencoder_checkpoints_steps,
    )
  )
  repr_fn, inv_repr_fn = processing_fns(
    variable_names=["target", "context"],
    repres=FLAGS.config.representation,
    max_len=FLAGS.config.representation.max_len,
  )
  train_iter, val_iter = get_train_iters(
    iter_key, FLAGS.config, FLAGS.workdir, repr_fn
  )
  state, _ = _new_train_state(init_key, FLAGS, FLAGS.config, teacher_params)
  mngr, ckpt_save_fn, *_ = get_checkpoint_manager(
    os.path.join(FLAGS.workdir, "checkpoints", run_name),
    FLAGS.config,
    FLAGS.config.model.to_dict(),
  )
  state, cmetrics, cstep = get_latest_train_state(mngr, state)

  pstep_fn = jax.pmap(
    latent_common.wrap_step(step_fn, target_encoder_fn, context_encoder_fn),
    axis_name="batch",
    in_axes=(0, 0, 0, None),
  )
  peval_fn = jax.pmap(
    latent_common.wrap_eval(eval_fn, target_encoder_fn, context_encoder_fn),
    axis_name="batch",
    in_axes=(0, 0, 0, None),
  )
  p_sample_fn = jax.pmap(
    get_sample_fn(
      sampling_fn,
      target_encoder_fn,
      context_encoder_fn,
      target_decoder_fn,
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
    range(cstep + 1, FLAGS.config.training.n_steps + 1), train_iter
  ):
    distillation_stage = (step - 1) // FLAGS.config.training.distillation_stages
    train_key, val_key, sample_key = jr.split(jr.fold_in(step_key, step), 3)
    if step == 1 and jax.process_index() == 0:
      logging.info(f"pbatch shape: {pbatch['target'].shape}")
      logging.info(
        f"pbatch quantile: {jnp.quantile(pbatch['target'], q=jnp.array([0.1, 0.25, 0.5, 0.75, 0.9]))}"
      )
    metrics, pstate = pstep_fn(
      jr.split(train_key, jax.device_count()),
      pstate,
      pbatch,
      distillation_stage,
    )
    train_metrics.append(metrics)
    is_first_step = step == 1
    is_last_step = step == FLAGS.config.training.n_steps
    is_first_or_last_step = is_first_step or is_last_step
    if (
      step % FLAGS.config.training.n_eval_frequency == 0
      or is_first_or_last_step
    ):
      val_metrics = []
      for val_idx, pbatch in zip(
        range(FLAGS.config.training.n_eval_batches), val_iter
      ):
        metrics = peval_fn(
          jr.split(jr.fold_in(val_key, val_idx), jax.device_count()),
          pstate,
          pbatch,
          distillation_stage,
        )
        val_metrics.append(metrics)
      summary = metrics_to_summary(train_metrics, val_metrics)
      train_metrics = []
      if jax.process_index() == 0:
        logging.info(
          f"step {step} train/val loss :{summary['train/loss']}/{summary['val/loss']}"
        )
      if (
        step % FLAGS.config.training.n_checkpointing_frequency == 0
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
      if FLAGS.usewand and jax.process_index() == 0:
        wandb.log(summary, step=step)
    if step % FLAGS.config.training.n_sampling_frequency == 0 or is_last_step:
      summaries = {}
      logging.info(f"sampling with {FLAGS.config.model.sampler.n_steps} steps")
      for sample_idx, pbatch in zip(
        range(FLAGS.config.training.n_sampling_batches), val_iter
      ):
        (_, specs_hat), _ = p_sample_fn(
          jr.split(jr.fold_in(sample_key, sample_idx), jax.device_count()),
          pstate,
          pbatch,
        )
        specs_hat = specs_hat.reshape((-1, *specs_hat.shape[2:]))
        specs = pbatch["target"].reshape((-1, *pbatch["target"].shape[2:]))
        signals = pbatch["signal"].reshape((-1, *pbatch["signal"].shape[2:]))
        signals_hat = inv_repr_fn(specs_hat)
        summaries = callback_summarize(
          step, callbacks, signals, signals_hat, summaries
        )
      if jax.process_index() == 0:
        plot_spectrograms(step, specs[:5], specs_hat[:5], FLAGS.usewand)
        callback_log(FLAGS.usewand, step, callbacks, summaries)
  logging.info("finished training")
