"""Jitted apply wrappers: model + params → callable."""

import jax


def dit_apply_fn(model, params):
  @jax.jit
  def fn(rngs, inputs, *, times, context, condition):
    y = model.apply(
      variables={"params": params},
      inputs=inputs,
      context=context,
      times=times * 999.0,
      condition=condition,
      is_training=False,
    )
    return y

  return fn


def vae_encoding_fn(model, params, return_z=True):
  @jax.jit
  def fn(rngs, inputs):
    z, (shift, scale) = model.apply(
      variables={"params": params},
      rngs=rngs,
      inputs=inputs,
      is_training=False,
      method=model.encode,
    )
    return z if return_z else shift

  return fn


def vae_decoding_fn(model, params):
  @jax.jit
  def fn(rngs, inputs):
    y = model.apply(
      variables={"params": params},
      inputs=inputs,
      is_training=False,
      method=model.decode,
    )
    return y

  return fn


def inception_apply_fn(model, params):
  @jax.jit
  def fn(inputs):
    y, embedding = model.apply(
      variables={"params": params},
      inputs=inputs,
      is_training=False,
    )
    return y, embedding

  return fn
