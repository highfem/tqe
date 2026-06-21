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
from latent_diffusion_experiment import get_sample_fn
from utils import (
  callback_log,
  callback_summarize,
  metrics_to_summary,
  plot_signals,
)

from tqe.nn import gan_pix2pix
from tqe.signal import processing_fns
from tqe.wgan import WGANConfig, WGANState, wgan


def get_model_and_matching_fns(config):
  if config.model.nn.name == "pix2pix":
    logging.info("using pix2pix")
    generator_model, critic_model = gan_pix2pix.make_models(config.model.nn)
  else:
    raise ValueError(f"unknown nn {config.model.nn.name!r}")
  matching_fns = wgan(
    generator_model.apply,
    critic_model.apply,
    WGANConfig(**config.model.gan.to_dict()),
    config.training.ema_rate,
  )
  return (generator_model, critic_model), matching_fns


def _new_train_state(rng_key, FLAGS, config):
  logging.info("initializing latent wgan")
  (generator_model, critic_model), *_ = get_model_and_matching_fns(config)
  (target_enc_fn, context_enc_fn), _ = latent_common.get_encoders_and_decoders(
    FLAGS.workdir,
    FLAGS.autoencoder_checkpoints,
    FLAGS.autoencoder_checkpoints_steps,
  )

  params_key, sample_key = jr.split(rng_key)
  batch = jnp.ones(
    (
      jax.device_count() * 2,
      *config.data.image_size,
      config.data.n_out_channels,
    )
  )
  condition = jnp.ones((jax.device_count() * 2, config.data.condition_length))

  if target_enc_fn is not None:
    targets = target_enc_fn({"sample": jr.PRNGKey(0)}, batch)
    context = context_enc_fn({"sample": jr.PRNGKey(0)}, batch)
    logging.info(f"pbatch shape target lowdim: {targets.shape}")
    logging.info(f"pbatch shape context lowdim: {context.shape}")
  else:
    targets = context = batch

  generator_variables = generator_model.init(
    {"params": params_key, "sample": sample_key},
    context=context,
    condition=condition,
    is_training=False,
  )
  critic_variables = critic_model.init(
    {"params": params_key, "sample": sample_key},
    inputs=targets,
    context=context,
    condition=condition,
    is_training=False,
  )
  n_params = jax.tree.map(lambda x: x.size, generator_variables["params"])
  n_params = jax.tree.leaves(n_params)
  n_params = jnp.sum(jnp.array(n_params))
  logging.info(f"total number of parameters of model: {n_params}")
  generator_state = train_state.new_ema_train_state(
    generator_variables, generator_model, config
  )
  critic_state = train_state.new_train_state(
    critic_variables, critic_model, config
  )
  return WGANState(critic=critic_state, generator=generator_state, step=0)


def train(rng_key, FLAGS, callbacks, run_name):
  iter_key, init_key, rng_key = jr.split(jr.PRNGKey(rng_key), 3)

  # loss and sampling functions
  _, matching_fns = get_model_and_matching_fns(FLAGS.config)
  (step_fn, eval_fn), sample_fn, _ = matching_fns
  # autoencoders
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

  # get initial state and restore from checkpoint if one exists
  init_state = _new_train_state(init_key, FLAGS, FLAGS.config)
  # two checkpoint managers for backward compat with existing checkpoints
  generator_mngr, generator_ckpt_save_fn, _ = get_checkpoint_manager(
    os.path.join(FLAGS.workdir, "checkpoints", run_name + "-generator"),
    FLAGS.config,
    FLAGS.config.model.to_dict(),
  )
  critic_mngr, critic_ckpt_save_fn, _ = get_checkpoint_manager(
    os.path.join(FLAGS.workdir, "checkpoints", run_name + "-critic"),
    FLAGS.config,
    FLAGS.config.model.to_dict(),
  )
  generator_state, _, cstep = get_latest_train_state(
    generator_mngr, init_state.generator
  )
  critic_state, *_ = get_latest_train_state(critic_mngr, init_state.critic)
  state = WGANState(critic=critic_state, generator=generator_state, step=0)

  # make everything distributed
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
      sample_fn,
      target_encoder_fn,
      context_encoder_fn,
      target_decoder_fn,
    ),
    axis_name="batch",
  )
  pstate = jax_utils.replicate(state)

  # train loop
  if jax.process_index() == 0:
    logging.info("training model")
    logging.info(f"starting/resuming training at step: {cstep}")
  step_key, rng_key = jr.split(rng_key)
  train_metrics = []
  for step, pbatch in zip(
    range(cstep + 1, FLAGS.config.training.n_steps + 1), train_iter
  ):
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
    )
    train_metrics.append(metrics)
    # define steps for checking
    is_first_step = step == 1
    is_last_step = step == FLAGS.config.training.n_steps
    is_first_or_last_step = is_first_step or is_last_step
    # do validation loops
    if (
      step % FLAGS.config.training.n_eval_frequency == 0
      or is_first_or_last_step
    ):
      val_metrics = []
      for val_idx, pbatch in zip(
        range(FLAGS.config.training.n_eval_batches), val_iter
      ):
        metrics, _ = peval_fn(
          jr.split(jr.fold_in(val_key, val_idx), jax.device_count()),
          pstate,
          pbatch,
        )
        val_metrics.append(metrics)
      summary = metrics_to_summary(train_metrics, val_metrics)
      train_metrics = []
      if jax.process_index() == 0:
        logging.info(
          f"step {step} "
          f"train/val critic loss :{summary['train/loss_c']}/{summary['val/loss_c']}, "
          f"train/val generator loss :{summary['train/loss_g']}/{summary['val/loss_g']}"
        )
      if (
        step % FLAGS.config.training.n_checkpointing_frequency == 0
        and jax.process_index() == 0
      ):
        unrep = jax_utils.unreplicate(pstate)
        critic_ckpt_save_fn(step, unrep.critic, summary)
        generator_ckpt_save_fn(step, unrep.generator, summary)
      if FLAGS.usewand and jax.process_index() == 0:
        wandb.log(summary, step=step)
    # reconstruct images
    if step % FLAGS.config.training.n_sampling_frequency == 0 or is_last_step:
      summaries = {}
      for sample_idx, pbatch in zip(
        range(FLAGS.config.training.n_sampling_batches), val_iter
      ):
        (specs_latents_hat, specs_hat), (specs_latent_train, specs_from_enc) = (
          p_sample_fn(
            jr.split(jr.fold_in(sample_key, sample_idx), jax.device_count()),
            pstate,
            pbatch,
          )
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
        specs = pbatch["target"].reshape((-1, *pbatch["target"].shape[2:]))
        signals = pbatch["signal"].reshape((-1, *pbatch["signal"].shape[2:]))
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
          FLAGS.usewand,
        )
        callback_log(FLAGS.usewand, step, callbacks, summaries)
  logging.info("finished training")
