"""Wandb logging, metrics summaries, plotting, pickle I/O, and env setup."""

import pickle

import jax
import numpy as np
import wandb
from absl import logging
from flax.training import common_utils
from jax.lib import xla_bridge
from matplotlib import pyplot as plt


def log_jax_env():
  logging.info("----- Checking JAX installation ----")
  logging.info(jax.devices())
  logging.info(jax.default_backend())
  logging.info(xla_bridge.get_backend().platform)
  logging.info("------------------------------------")


def save_pickle(outfile, obj):
  with open(outfile, "wb") as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(outfile):
  with open(outfile, "rb") as handle:
    x = pickle.load(handle)
  return x


def callback_log(use_wandb, step, callbacks, summaries):
  logging.info("logging images to wand")
  metrics, plots = {}, {}
  for callback in callbacks:
    summary = summaries[callback.name]
    metrics[callback.name] = callback.compute(
      summary["targets"], summary["preds"]
    )
    if callback.plot is not None:
      fig = callback.plot(summary["targets"], summary["preds"])
      plots[callback.name + " (image)"] = fig
  if use_wandb:
    wandb.log(metrics, step=step)
    for name, fig in plots.items():
      wandb.log({f"{name}": wandb.Image(fig)}, step=step)
    logging.info("done")


def callback_summarize(step, callbacks, signals, signals_hat, summaries):
  # subset, since TF invert_spectrogram doesnt manage to do it correctly
  # axis 1 is the signal axis
  min_len = np.minimum(signals.shape[1], signals_hat.shape[1])
  signals, signals_hat = signals[:, :min_len, :], signals_hat[:, :min_len, :]
  for callback in callbacks:
    target_stats, pred_stats = callback.summarize(signals, signals_hat)
    if callback.name not in summaries:
      summaries[callback.name] = {"targets": [], "preds": []}
    summaries[callback.name]["targets"].append(target_stats)
    summaries[callback.name]["preds"].append(pred_stats)
  return summaries


def plot_signals(
  step,
  signals,
  signals_from_encoder,
  signals_hat,
  specs,
  specs_from_enc,
  specs_hat,
  specs_latents,
  specs_latents_hat,
  use_wandb,
):
  if signals_from_encoder is None:
    fig, axes = plt.subplots(figsize=(20, 6), ncols=2, nrows=5)
  else:
    fig, axes = plt.subplots(figsize=(20, 9), ncols=3, nrows=5)
  leng = len(signals)
  for i in range(leng):
    axes[i, 0].set_title("Original signal (raw data)")
    axes[i, 0].plot(signals[i], alpha=0.5)
    axes[i, 1].set_title("Modelled signal")
    axes[i, 1].plot(signals_hat[i], alpha=0.5)
    if signals_from_encoder is not None:
      axes[i, 2].set_title("Reconstructed signal with autoencoder")
      axes[i, 2].plot(signals_from_encoder[i], alpha=0.5)
  plt.tight_layout()
  plt.close()
  if use_wandb:
    wandb.log({"Signals": wandb.Image(fig)}, step=step)

  if specs_from_enc is None:
    fig, axes = plt.subplots(figsize=(20, 6), ncols=2, nrows=5)
  else:
    fig, axes = plt.subplots(figsize=(20, 15), ncols=5, nrows=5)
  for i in range(leng):
    axes[i, 0].set_title("True spectrogram")
    axes[i, 0].imshow(specs[i, :, :, 0], alpha=0.5)
    axes[i, 1].set_title("Modelled spectrogram")
    axes[i, 1].imshow(specs_hat[i, :, :, 0], alpha=0.5)
    if specs_from_enc is not None:
      axes[i, 2].set_title("Recovered spectrogram")
      axes[i, 2].imshow(specs_from_enc[i, :, :, 0], alpha=0.5)
      axes[i, 3].set_title("Modelled latent spectrogram")
      axes[i, 3].imshow(specs_latents_hat[i, :, :, 0], alpha=0.5)
      axes[i, 4].set_title("True latent spectrogram (train data)")
      axes[i, 4].imshow(specs_latents[i, :, :, 0], alpha=0.5)
  plt.tight_layout()
  plt.close()
  if use_wandb:
    wandb.log({"Spectrograms": wandb.Image(fig)}, step=step)

  if specs_latents is not None:
    n_channels = specs_latents.shape[-1]
    fig, axes = plt.subplots(figsize=(20, 24), ncols=n_channels, nrows=5)
    for i in range(leng):
      for j in range(n_channels):
        axes[i, j].imshow(specs_latents[i, :, :, j], alpha=0.5)
    plt.tight_layout()
    plt.close()
    if use_wandb:
      wandb.log(
        {"All latent spectrogram channels": wandb.Image(fig)}, step=step
      )

  return fig


def plot_spectrograms(step, specs, specs_hat, use_wandb):
  fig, axes = plt.subplots(figsize=(20, 6), ncols=2, nrows=5)
  for i in range(len(specs)):
    axes[i, 0].set_title("True spectrogram")
    axes[i, 0].imshow(specs[i, :, :, 0], alpha=0.5)
    axes[i, 1].set_title("Modelled spectrogram")
    axes[i, 1].imshow(specs_hat[i, :, :, 0], alpha=0.5)
  plt.tight_layout()
  plt.close()
  if use_wandb:
    wandb.log({"Spectrograms": wandb.Image(fig)}, step=step)


def metrics_to_summary(train_metrics, val_metrics):
  train_metrics = common_utils.get_metrics(train_metrics)
  val_metrics = common_utils.get_metrics(val_metrics)
  train_summary = {
    f"train/{k}": v
    for k, v in jax.tree_map(lambda x: float(x.mean()), train_metrics).items()
  }
  val_summary = {
    f"val/{k}": v
    for k, v in jax.tree_map(lambda x: float(x.mean()), val_metrics).items()
  }
  return train_summary | val_summary
