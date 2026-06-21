import os

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from flax import jax_utils
from flax.training import common_utils
from jax import numpy as jnp
from jax import random as jr


def get_cardinality(dataset, split="train", outpath: str = None):
  if isinstance(split, str):
    split = [split]
  exp_ds = tfds.load(
    dataset, try_gcs=False, data_dir=outpath, split=split, shuffle_files=True
  )
  cards = dict(zip(split, [int(ds.cardinality().numpy()) for ds in exp_ds]))
  return cards


def data_loaders(
  rng_key,
  config,
  dataset,
  representation_fn,
  split="train",
  outpath: str = None,
  repeat=None,
  batch_size_per_gpu=None,
  drop_remainder=True,
):
  if isinstance(split, str):
    split = [split]
  exp_ds = tfds.load(
    dataset, try_gcs=False, data_dir=outpath, split=split, shuffle_files=True
  )
  cards = dict(zip(split, [int(ds.cardinality().numpy()) for ds in exp_ds]))
  exp_ds = dict(zip(split, exp_ds))
  for name, itr in exp_ds.items():
    split_key, rng_key = jr.split(rng_key)
    exp_ds[name] = _as_batched_numpy_iter(
      split_key,
      itr,
      config,
      representation_fn,
      repeat=repeat,
      batch_size_per_gpu=batch_size_per_gpu,
      drop_remainder=drop_remainder,
    )
  return exp_ds, cards


def _as_batched_numpy_iter(
  rng_key,
  itr,
  config,
  representation_fn,
  repeat=None,
  batch_size_per_gpu=None,
  drop_remainder=True,
):
  max_int32 = jnp.iinfo(jnp.int32).max
  seed = jr.randint(rng_key, shape=(), minval=0, maxval=max_int32)

  batch_size = config.batch_size
  if batch_size_per_gpu is not None:
    batch_size = batch_size_per_gpu * jax.device_count()

  ds = (
    itr.repeat(count=repeat)
    .shuffle(
      config.buffer_size,
      reshuffle_each_iteration=config.do_reshuffle,
      seed=int(seed),
    )
    .batch(batch_size, drop_remainder=drop_remainder)
    .map(representation_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .prefetch(config.prefetch_size)
    .as_numpy_iterator()
  )
  return ds


def get_train_iters(rng_key, config, workdir, representation_fn, as_dict=False):
  val_split = config.training.percentage_data_as_validation_set
  train_size = int(100 * (1 - val_split))
  train_size_str, val_size_str = (
    f"train[:{train_size}%]",
    f"train[{train_size}%:]",
  )
  itrs, _ = data_loaders(
    rng_key=rng_key,
    config=config.training,
    dataset=config.data.dataset,
    split=[train_size_str, val_size_str],
    representation_fn=representation_fn,
    outpath=os.path.join(workdir, "data"),
  )
  for k, itr in itrs.items():
    itr = map(common_utils.shard, itr)
    itr = jax_utils.prefetch_to_device(itr, 2)
    itrs[k] = itr

  if as_dict:
    return itrs
  return itrs[train_size_str], itrs[val_size_str]


def get_test_iter(
  rng_key,
  config,
  workdir,
  representation_fn,
  as_dict=False,
  batch_size_per_gpu=None,
  repeat=None,
):
  logging.info("loading test data set")
  itrs, cards = data_loaders(
    rng_key=rng_key,
    config=config.training,
    dataset=config.data.dataset,
    split=["test"],
    representation_fn=representation_fn,
    repeat=repeat,
    outpath=os.path.join(workdir, "data"),
    batch_size_per_gpu=batch_size_per_gpu,
    drop_remainder=False,
  )
  logging.info(
    f"loaded data sets '{'/'.join(itrs.keys())}' "
    f"with sizes '{'/'.join(map(str, cards.values()))}'"
  )

  def make_device_batchable(batch):
    n = batch["context"].shape[0]
    n = n - n % jax.device_count()
    for k, v in batch.items():
      batch[k] = v[:n]
    return batch

  for k, itr in itrs.items():
    itr = map(make_device_batchable, itr)
    itr = map(common_utils.shard, itr)
    itr = jax_utils.prefetch_to_device(itr, 2)
    itrs[k] = itr

  if as_dict:
    return itrs, cards
  return itrs["test"], cards["test"]
