"""Orbax checkpoint save/restore and model checkpoint loading."""

import os

import numpy as np
import orbax.checkpoint
import train_state
from absl import logging
from flax.training import orbax_utils
from ml_collections import ConfigDict
from train_state import _restore_state
from utils import load_pickle, save_pickle

from tqe.nn import dit, inception, vae


def get_checkpoint_manager(
  outfolder, config, model_config, criterion="val/loss"
):
  options = orbax.checkpoint.CheckpointManagerOptions(
    max_to_keep=config.training.checkpoints.max_to_keep,
    save_interval_steps=config.training.checkpoints.save_interval_steps,
    create=True,
  )
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  checkpoint_manager = orbax.checkpoint.CheckpointManager(
    os.path.join(outfolder, "last"),
    checkpointer,
    options,
  )
  save_pickle(os.path.join(outfolder, "config.pkl"), model_config)

  def save_fn(step, ckpt, metrics, best=False):
    ckpt = {"state": ckpt, "metrics": metrics, "config": model_config}
    save_args = orbax_utils.save_args_from_target(ckpt)
    if not best:
      try:
        checkpoint_manager.save(
          step=step,
          items=ckpt,
          save_kwargs={"save_args": save_args},
          metrics=metrics,
          force=True,
        )
        checkpoint_manager.wait_until_finished()
      except Exception:
        logging.exception("checkpoint save failed; resuming training")
    else:
      checkpointer.save(os.path.join(outfolder, "best"), ckpt, force=True)

  def restore_last_fn():
    return checkpoint_manager.restore(checkpoint_manager.latest_step())

  return checkpoint_manager, save_fn, restore_last_fn


def get_latest_train_state(mngr, state):
  try:
    logging.info("trying to restore train state")
    state, metrics = train_state.restore_train_state(mngr, state, "latest")
    logging.info("successfully restored train state")
    return state, metrics, mngr.latest_step()
  except Exception:
    logging.info("could not restore train state. starting from scratch")
    metrics = {"train/loss": np.inf, "val/loss": np.inf, "val/best": np.inf}
  return state, metrics, 0


def get_best_train_state(path, state):
  logging.info("trying to restore train state")
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  ckpt = checkpointer.restore(os.path.join(path, "best"))
  state, _ = _restore_state(state, ckpt)
  return state, _


def load_config(pth) -> dict:
  return load_pickle(os.path.join(pth, "config.pkl"))


def load_params(pth, step, *, use_ema: bool = False):
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  if isinstance(step, str) and step in ("latest", "last"):
    path = os.path.join(pth, "last")
    checkpoint_manager = orbax.checkpoint.CheckpointManager(path, checkpointer)
    ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())
  elif isinstance(step, str) and step == "best":
    path = os.path.join(pth, "best")
    ckpt = checkpointer.restore(path)
  else:
    raise ValueError(f"unsupported checkpoint step {step!r}")
  state_key = "ema_params" if use_ema else "params"
  return ckpt["state"][state_key]


def load_dit_with_ema_weights(pth, step):
  cfg = load_config(pth)
  model = dit.make_model(ConfigDict(cfg).nn.dit_score_net)
  params = load_params(pth, step, use_ema=True)
  return model, params


def load_vae(pth, step):
  cfg = load_config(pth)
  model = vae.make_model(ConfigDict(cfg).nn)
  params = load_params(pth, step, use_ema=False)
  return model, params


def load_inception(pth, step):
  cfg = load_config(pth)
  model = inception.make_model(ConfigDict(cfg).nn)
  params = load_params(pth, step, use_ema=False)
  return model, params
