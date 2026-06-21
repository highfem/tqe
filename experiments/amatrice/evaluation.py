import os

import h5py
import jax
import latent_consistency_distillation_experiment
import latent_diffusion_experiment
import latent_wgan_experiment
import model_fns
from absl import logging
from checkpoint import (
  get_best_train_state,
  get_checkpoint_manager,
  get_latest_train_state,
  load_inception,
)
from data import get_cardinality, get_test_iter
from flax import jax_utils
from jax import random as jr
from latent_common import get_encoders_and_decoders
from latent_diffusion_experiment import get_sample_fn

from tqe.signal import processing_fns


def create_generator_and_repr(iter_key, FLAGS):
  repr_fn, inv_repr_fn = processing_fns(
    variable_names=["target", "context"],
    repres=FLAGS.config.representation,
    max_len=FLAGS.config.representation.max_len,
    return_context_signal=True,
  )
  test_iter_cardinality = get_cardinality(
    dataset=FLAGS.config.data.dataset,
    split=["test"],
    outpath=os.path.join(FLAGS.workdir, "data"),
  )["test"]
  batch_size = FLAGS["batch-size-per-device"].value * jax.device_count()

  test_iter, _ = get_test_iter(
    iter_key,
    FLAGS.config,
    FLAGS.workdir,
    repr_fn,
    batch_size_per_gpu=FLAGS["batch-size-per-device"].value,
    repeat=1,
  )
  generator = zip(
    range(
      0,
      test_iter_cardinality,
      batch_size,
    ),
    test_iter,
  )

  return generator, (test_iter_cardinality, batch_size), (repr_fn, inv_repr_fn)


def get_dev_count_divisible_n_samples(cardinality, batch_size):
  # number of full batches and remaining samples
  full_batches, remainder = divmod(cardinality, batch_size)
  # adjust leftover samples so total batches are divisible by number of devices
  adjustment = remainder % jax.device_count()
  leftover = remainder - adjustment
  # compute the final cardinaliyu
  return full_batches * batch_size + leftover


def get_classifier(FLAGS):
  model, params = get_classifier_checkpoint(FLAGS)
  classifier = model_fns.inception_apply_fn(model, params)
  return classifier


def get_classifier_checkpoint(FLAGS):
  checkpoint_pth = FLAGS.classifier_checkpoint
  logging.info(f"initializing classifier {checkpoint_pth}")
  pth = os.path.join(FLAGS.workdir, "checkpoints", checkpoint_pth)
  return load_inception(pth, "best")


def create_datasets(f, FLAGS, total_num_samples):
  f_ts = f.create_dataset(
    "target",
    (total_num_samples, FLAGS.config.representation.max_len, 3),
  )
  f_tsh = f.create_dataset(
    "target_hat",
    (total_num_samples, FLAGS.config.representation.max_len, 3),
  )
  f_ts_emb = f.create_dataset(
    "target_classifier_embedding",
    (total_num_samples, 1600),
  )
  f_tsh_emb = f.create_dataset(
    "target_hat_classifier_embedding",
    (total_num_samples, 1600),
  )
  f_ts_preds = f.create_dataset(
    "target_preds",
    (total_num_samples, 20),
  )
  f_tsh_preds = f.create_dataset(
    "target_hat_preds",
    (total_num_samples, 20),
  )
  f_cs = f.create_dataset(
    "context",
    (total_num_samples, FLAGS.config.representation.max_len, 3),
  )
  f_metas = f.create_dataset(
    "meta", (total_num_samples, FLAGS.config.data.condition_length)
  )
  return (
    (f_ts, f_ts_emb, f_ts_preds),
    (f_tsh, f_tsh_emb, f_tsh_preds),
    f_cs,
    f_metas,
  )


def get_samples_sizes(FLAGS, cardinality, batch_size):
  # make total_num_samples divisible by number of devices
  total_num_samples = get_dev_count_divisible_n_samples(cardinality, batch_size)
  return total_num_samples


def get_sampling_fn(FLAGS):
  if FLAGS.config.model_type == "latent_diffusion":
    _, matching_fns = latent_diffusion_experiment.get_model_and_matching_fns(
      FLAGS.config
    )
    _, sampling_fn, _ = matching_fns
  elif FLAGS.config.model_type == "latent_wgan":
    _, matching_fns = latent_wgan_experiment.get_model_and_matching_fns(
      FLAGS.config
    )
    _, sampling_fn, _ = matching_fns
  elif FLAGS.config.model_type == "latent_consistency_distillation":
    _, matching_fns = (
      latent_consistency_distillation_experiment.get_model_and_matching_fns(
        FLAGS.config
      )
    )
    _, sampling_fn, _ = matching_fns
  else:
    raise ValueError(f"unknown model_type {FLAGS.config.model_type!r}")
  return sampling_fn


def _new_train_state(init_key, FLAGS):
  if FLAGS.config.model_type in [
    "latent_diffusion",
    "latent_consistency_distillation",
  ]:
    state, model = latent_diffusion_experiment._new_train_state(
      init_key,
      FLAGS.config,
      FLAGS.workdir,
      FLAGS.autoencoder_checkpoints,
      FLAGS.autoencoder_checkpoints_steps,
    )
  elif FLAGS.config.model_type == "latent_wgan":
    state = latent_wgan_experiment._new_train_state(
      init_key, FLAGS, FLAGS.config
    )
    return state, None
  else:
    raise ValueError(f"unknown model_type {FLAGS.config.model_type!r}")

  return state, model


def _get_state(rng_key, FLAGS):
  state, _ = _new_train_state(rng_key, FLAGS)
  if FLAGS.config.model_type in [
    "latent_wgan",
    "latent_consistency_distillation",
  ]:
    mngr, *_ = get_checkpoint_manager(
      os.path.join(FLAGS.workdir, "checkpoints", FLAGS.model_checkpoint),
      FLAGS.config,
      FLAGS.config.model.to_dict(),
    )
    if FLAGS.config.model_type == "latent_wgan":
      state = state.generator
    state, *_ = get_latest_train_state(mngr, state)
  else:
    state, _ = get_best_train_state(
      os.path.join(FLAGS.workdir, "checkpoints", FLAGS.model_checkpoint), state
    )
  return state


def evaluate(rng_key, FLAGS):
  iter_key, init_key, rng_key = jr.split(jr.PRNGKey(rng_key), 3)
  # loss and sampling functions
  sampling_fn = get_sampling_fn(FLAGS)
  # autoencoders
  (target_encoder_fn, context_encoder_fn), (target_decoder_fn, _) = (
    get_encoders_and_decoders(
      FLAGS.workdir,
      FLAGS.autoencoder_checkpoints,
      FLAGS.autoencoder_checkpoints_steps,
    )
  )
  classifier_fn = get_classifier(FLAGS)
  # construct a parallel sampling function
  p_sample_fn = jax.pmap(
    get_sample_fn(
      sampling_fn,
      target_encoder_fn,
      context_encoder_fn,
      target_decoder_fn,
    ),
    axis_name="batch",
  )
  state = _get_state(init_key, FLAGS)
  pstate = jax_utils.replicate(state)
  # get data iterator
  test_iter, (cardinality, batch_size), (repr_fn, inv_repr_fn) = (
    create_generator_and_repr(iter_key, FLAGS)
  )
  total_num_samples = get_samples_sizes(FLAGS, cardinality, batch_size)

  # make predictions
  with h5py.File(FLAGS.outfile, "w") as f:
    (
      (f_ts, f_ts_emb, f_ts_preds),
      (f_tsh, f_tsh_emb, f_tsh_preds),
      f_cs,
      f_metas,
    ) = create_datasets(f, FLAGS, total_num_samples)

    curr_starting_idx = 0
    for idx, pbatch in test_iter:
      sample_key, rng_key = jr.split(rng_key)
      (_, specs_hat), _ = p_sample_fn(
        jr.split(sample_key, jax.device_count()),
        pstate,
        pbatch,
      )
      specs_hat = specs_hat.reshape((-1, *specs_hat.shape[2:]))
      signals_hat = inv_repr_fn(specs_hat)
      context = pbatch["context_signal"].reshape(
        (-1, *pbatch["context_signal"].shape[2:])
      )
      metas = pbatch["meta"].reshape((-1, pbatch["meta"].shape[-1]))
      signals = pbatch["signal"].reshape((-1, *pbatch["signal"].shape[2:]))
      specs = pbatch["target"].reshape((-1, *pbatch["target"].shape[2:]))
      target_preds, target_embedding = classifier_fn(specs)
      target_hat_preds, target_hat_embedding = classifier_fn(specs_hat)
      min_leng = min(context.shape[1], signals_hat.shape[1], signals.shape[1])
      if jax.process_index() == 0:
        writable_idxs = context.shape[0]
        end_idxs = curr_starting_idx + writable_idxs
        f_ts[curr_starting_idx:end_idxs, :min_leng, :] = signals[
          :writable_idxs, :min_leng, :
        ]
        f_ts_emb[curr_starting_idx:end_idxs] = target_embedding[:writable_idxs]
        f_ts_preds[curr_starting_idx:end_idxs] = target_preds[:writable_idxs]
        f_tsh[curr_starting_idx:end_idxs, :min_leng, :] = signals_hat[
          :writable_idxs, :min_leng, :
        ]
        f_tsh_emb[curr_starting_idx:end_idxs] = target_hat_embedding[
          :writable_idxs
        ]
        f_tsh_preds[curr_starting_idx:end_idxs] = target_hat_preds[
          :writable_idxs
        ]
        f_cs[curr_starting_idx:end_idxs, :min_leng, :] = context[
          :writable_idxs, :min_leng, :
        ]
        f_metas[curr_starting_idx:end_idxs] = metas[:writable_idxs]
        curr_starting_idx += writable_idxs
  logging.info("done!")
