import os

import checkpoint
import jax
import model_fns
from absl import logging
from jax import numpy as jnp
from jax import random as jr


def encode_batch(rng_key, batch, target_encoder_fn, context_encoder_fn):
  """Map experiment batch keys to generative objective batch keys."""
  _, target_key, context_key = jr.split(rng_key, 3)
  inputs = batch["target"]
  context = batch["context"]
  if target_encoder_fn is not None:
    inputs = target_encoder_fn({"sample": target_key}, batch["target"])
  if context_encoder_fn is not None:
    context = context_encoder_fn({"sample": context_key}, batch["context"])
  encoded = {"inputs": inputs, "context": context}
  if "meta" in batch:
    encoded["condition"] = batch["meta"]
  return encoded


def wrap_step(step_fn, target_encoder_fn, context_encoder_fn):
  @jax.jit
  def fn(rng_key, state, batch, *args):
    return step_fn(
      rng_key,
      state,
      encode_batch(rng_key, batch, target_encoder_fn, context_encoder_fn),
      *args,
    )

  return fn


def wrap_eval(eval_fn, target_encoder_fn, context_encoder_fn):
  @jax.jit
  def fn(rng_key, state, batch, *args):
    return eval_fn(
      rng_key,
      state,
      encode_batch(rng_key, batch, target_encoder_fn, context_encoder_fn),
      *args,
    )

  return fn


def get_autoencoder_checkpoints(
  workdir, autoencoder_checkpoints, autoencoder_checkpoints_steps=None
):
  if autoencoder_checkpoints is None:
    logging.info("NOT initializing autoencoders")
    return None, None

  steps = autoencoder_checkpoints_steps
  if steps is None:
    steps = ["best"] * len(autoencoder_checkpoints)

  models, params = {}, {}
  for checkpoint_pth, checkpoint_step in zip(autoencoder_checkpoints, steps):
    logging.info(f"initializing autoencoder {checkpoint_pth}")
    pth = os.path.join(workdir, "checkpoints", checkpoint_pth)
    if "context" in checkpoint_pth:
      model_name = "context"
    elif "target" in checkpoint_pth:
      model_name = "target"
    else:
      raise ValueError(
        f"checkpoint path {checkpoint_pth!r} must contain 'context' or 'target'"
      )
    models[model_name], params[model_name] = checkpoint.load_vae(
      pth, checkpoint_step
    )
  return models, params


def get_encoders_and_decoders(
  workdir, autoencoder_checkpoints, autoencoder_checkpoints_steps=None
):
  models, params = get_autoencoder_checkpoints(
    workdir, autoencoder_checkpoints, autoencoder_checkpoints_steps
  )
  if models is None:
    return (None, None), (None, None)

  n_params = jax.tree.map(lambda x: x.size, params["target"])
  n_params = jnp.sum(jnp.array(jax.tree.leaves(n_params)))
  logging.info(f"total number of parameters of target encoder: {n_params}")

  target_enc = model_fns.vae_encoding_fn(models["target"], params["target"])
  context_enc = model_fns.vae_encoding_fn(models["context"], params["context"])
  target_dec = model_fns.vae_decoding_fn(models["target"], params["target"])
  context_dec = model_fns.vae_decoding_fn(models["context"], params["context"])
  return (target_enc, context_enc), (target_dec, context_dec)
