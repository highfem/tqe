"""Compose representation transforms for batches."""

from collections.abc import Callable

from absl import logging

from tqe.signal.log_spectrogram import log_spectrogram_fns
from tqe.signal.low_pass_filter import low_pass_filter_fns

_TRANSFORMS = {
  "log_spectrogram": log_spectrogram_fns,
  "low_pass_filter": low_pass_filter_fns,
}


def _lookup(name: str) -> Callable:
  if name not in _TRANSFORMS:
    raise ValueError(
      f"unknown transform '{name}'; available: {sorted(_TRANSFORMS)}"
    )
  return _TRANSFORMS[name]


def processing_fns(
  variable_names, repres, max_len, return_context_signal=False
):
  """Returns forward and inverse batch transforms for a representation stack.

  Args:
    variable_names: Batch key (``str``) or pair of keys (``list``) to encode.
    repres: Representation config with a list of transforms to apply.
    max_len: Maximum number of time samples to retain per waveform.
    return_context_signal: If True, keep raw context waveforms when encoding
      multiple variables.

  Returns:
    A ``(forward_fn, inverse_fn)`` tuple. ``forward_fn`` maps a raw batch dict
    to an encoded batch dict; ``inverse_fn`` maps encoded data back to waveforms.
  """
  forward_fns, inverse_fns = _build_transform_chain(repres)

  def _apply_chain(x):
    for f in forward_fns:
      x = f(x)
    return x

  def _invert_chain(x):
    for f in reversed(inverse_fns):
      x = f(x)
    return x

  if isinstance(variable_names, str):
    logging.info(f"creating repr for '{variable_names}'")

    def forward_fn(batch):
      signal = batch[variable_names][:, :max_len, :]
      return {
        "inputs": _apply_chain(signal),
        "signal": signal,
        "meta": batch["meta"],
      }

  elif isinstance(variable_names, list):
    logging.info(f"creating repr for {variable_names}")

    def forward_fn(batch):
      batch["signal"] = batch["target"][:, :max_len, :]
      if return_context_signal:
        batch["context_signal"] = batch["context"][:, :max_len, :]
      for key in variable_names:
        batch[key] = _apply_chain(batch[key][:, :max_len, :])
      return batch

  else:
    raise ValueError(
      f"variable_names must be a str or list, got {type(variable_names).__name__!r}"
    )

  return forward_fn, lambda x: _invert_chain(x)


def _build_transform_chain(repres) -> tuple[list[Callable], list[Callable]]:
  """Builds paired forward/inverse transform lists from the representations config."""
  forward_fns, inverse_fns = [], []
  for repre in repres.representations:
    fwd, inv = _lookup(repre.name)(**repre.params)
    forward_fns.append(fwd)
    inverse_fns.append(inv)
  return forward_fns, inverse_fns
