"""CLI for generating synthetic seismic waveforms from a pretrained latent DDM."""

from __future__ import annotations

import argparse
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import jax
import numpy as np
import orbax.checkpoint
import pooch
from absl import logging
from einops import rearrange
from flax import jax_utils
from flax.training import common_utils
from jax import numpy as jnp
from jax import random as jr
from ml_collections import ConfigDict

from tqe.denoising_diffusion import EDMParameterization, denoising_diffusion
from tqe.flow_matching import FlowMatchingConfig, flow_matching
from tqe.nn import dit, vae
from tqe.signal import processing_fns

_DESCRIPTION = """\
Generate synthetic seismic waveforms from a pretrained latent DDM.

The first run downloads pretrained weights from Zenodo to
~/.cache/tqe/weights/.  Override the cache location with $TQE_CACHE_DIR.

Input HDF5 datasets:
  context             float32 (N, L, 3)  three-component context waveforms
                      sampled at 100 Hz, L >= 8161 samples
                      ((N, 3, L) channel-first is accepted and transposed)
  magnitude, vs30, station_longitude, station_latitude,
  station_elevation, hypocentre_depth, hypocentre_longitude,
  hypocentre_latitude
                      float32 (N,)       per-sample scalar metadata

Output HDF5 datasets:
  target_hat          float32 (N × n_per_entry, max_len, 3)
  context             float32 (N × n_per_entry, max_len, 3)
  meta                float32 (N × n_per_entry, 8)
"""

# ---------------------------------------------------------------------------
# Zenodo weight registry
# ---------------------------------------------------------------------------

_ZENODO_BASE = "https://zenodo.org/records/23107915/files"
_WEIGHTS: dict[str, str] = {
  "context_encoder.tar.gz": (
    "sha256:f710323c80ff471d741df79cc108f0d5bba923af322eb98899ed38a1b167d083"
  ),
  "target_encoder.tar.gz": (
    "sha256:3c75aaa99ae1e439d6ef3bda96cf95aae647e6c064338f64594703cff33b4974"
  ),
  "latent_diffusion.tar.gz": (
    "sha256:44b857bccfbd59a4c3e04a6b468f685e809df1114d544cf63c7598b6f3a08b39"
  ),
}
_CACHE_DIR = (
  Path(os.environ.get("TQE_CACHE_DIR", str(Path.home() / ".cache" / "tqe")))
  / "weights"
)


def get_weights(cache_dir: Path | None = None) -> Path:
  """Download and extract pretrained weights to the local cache directory.

  Weights are cached; subsequent calls skip the download if checksums match.

  Args:
    cache_dir: Override for the default cache location (``$TQE_CACHE_DIR``
      or ``~/.cache/tqe/weights``).

  Returns:
    Path to the directory containing the extracted checkpoint subdirectories.
  """
  dest = cache_dir or _CACHE_DIR
  dest.mkdir(parents=True, exist_ok=True)
  for name, checksum in _WEIGHTS.items():
    pooch.retrieve(
      url=f"{_ZENODO_BASE}/{name}?download=1",
      known_hash=checksum,
      fname=name,
      path=str(dest),
      processor=pooch.Untar(extract_dir=str(dest)),
      progressbar=True,
    )
  return dest


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_pickle(path: Path) -> dict:
  with open(path, "rb") as f:
    return pickle.load(f)  # noqa: S301


def _restore_args(metadata):
  if isinstance(metadata, orbax.checkpoint.metadata.StringMetadata):
    return orbax.checkpoint.RestoreArgs()
  return orbax.checkpoint.RestoreArgs(restore_type=np.ndarray)


def _restore_best(checkpoint_dir: Path) -> dict:
  # Checkpoints record the device they were saved on (cuda:0 for the
  # pretrained weights); restoring as NumPy arrays loads them on any machine.
  path = str(checkpoint_dir / "best")
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  restore_args = jax.tree.map(_restore_args, checkpointer.metadata(path))
  return checkpointer.restore(path, restore_args=restore_args)


def load_vae(checkpoint_dir: Path) -> tuple:
  """Load a VAE model and its params from a checkpoint directory.

  Args:
    checkpoint_dir: Directory containing ``config.pkl`` and ``best/``.

  Returns:
    ``(model, params)`` ready for inference.
  """
  cfg = ConfigDict(_load_pickle(checkpoint_dir / "config.pkl"))
  model = vae.make_model(cfg.nn)
  params = _restore_best(checkpoint_dir)["state"]["params"]
  return model, params


def load_ddm(checkpoint_dir: Path) -> tuple:
  """Load a DiT score model with EMA weights and its full generation config.

  Args:
    checkpoint_dir: Directory containing ``config.pkl``, ``full_config.pkl``,
      and ``best/``.

  Returns:
    ``(model, full_cfg, ema_params)`` where ``full_cfg`` is a ``ConfigDict``
    of the training config; it must contain ``model``, ``representation``,
    and ``training.ema_rate``.

  Raises:
    FileNotFoundError: If ``full_config.pkl`` is absent.
  """
  full_cfg_path = checkpoint_dir / "full_config.pkl"
  if not full_cfg_path.exists():
    raise FileNotFoundError(
      f"full_config.pkl not found in {checkpoint_dir}. "
      "The latent_diffusion archive must include full_config.pkl: the "
      "training config (get_config().to_dict() without 'callbacks') with "
      "keys 'model', 'representation', and 'training.ema_rate'."
    )
  cfg = ConfigDict(_load_pickle(checkpoint_dir / "config.pkl"))
  full_cfg = ConfigDict(_load_pickle(full_cfg_path))
  model = dit.make_model(cfg.nn.dit_score_net)
  ema_params = _restore_best(checkpoint_dir)["state"]["ema_params"]
  return model, full_cfg, ema_params


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _flow_matching_sample_fn(apply_fn, n_steps, ema_rate):
  config = FlowMatchingConfig(n_sampling_steps=n_steps)
  return flow_matching(apply_fn, config, ema_rate).sample_fn


def _edm_sample_fn(apply_fn, n_steps, ema_rate):
  config = EDMParameterization(n_sampling_steps=n_steps)
  return denoising_diffusion(apply_fn, config, ema_rate).sample_fn


_SAMPLERS: dict[str, Callable[[Callable, int, float], Callable]] = {
  "flow_matching": _flow_matching_sample_fn,
  "edm": _edm_sample_fn,
}


def _make_sample_fn(
  name: str, apply_fn: Callable, n_steps: int, ema_rate: float
) -> Callable:
  """Builds the sampler for the objective the DiT was trained with.

  Args:
    name: Objective name stored in the checkpoint config (``model.name``).
    apply_fn: Flax ``apply`` of the score network.
    n_steps: Number of sampling steps.
    ema_rate: EMA decay rate; the objective factories require it, sampling
      does not use it.

  Returns:
    ``sample_fn(rng_key, state, context, condition)``.

  Raises:
    ValueError: If ``name`` is not a known objective.
  """
  if name not in _SAMPLERS:
    raise ValueError(
      f"unknown objective '{name}'; available: {sorted(_SAMPLERS)}"
    )
  return _SAMPLERS[name](apply_fn, n_steps, ema_rate)


class _InferenceState(NamedTuple):
  ema_params: Any


class GenerationPipeline(NamedTuple):
  """Loaded models and transforms ready for inference.

  Attributes:
    p_sample_fn: pmapped ``(rng_keys, pstate, batch) -> target_specs``.
    pstate: EMA params replicated across devices.
    repr_fn: Waveform-to-spectrogram transform.
    inv_repr_fn: Spectrogram-to-waveform inverse transform.
    max_len: Maximum waveform length in samples.
    condition_keys: Ordered metadata field names expected in the input HDF5.
  """

  p_sample_fn: Callable
  pstate: _InferenceState
  repr_fn: Callable
  inv_repr_fn: Callable
  max_len: int
  condition_keys: tuple[str, ...]


def build_pipeline(weights_dir: Path) -> GenerationPipeline:
  """Load pretrained models and assemble the generation pipeline.

  Args:
    weights_dir: Directory containing ``context_encoder/``, ``target_encoder/``,
      and ``latent_diffusion/`` checkpoint subdirectories.

  Returns:
    A :class:`GenerationPipeline` ready to pass to :func:`generate`.
  """
  logging.info("loading context encoder")
  ctx_model, ctx_params = load_vae(weights_dir / "context_encoder")
  logging.info("loading target encoder")
  tgt_model, tgt_params = load_vae(weights_dir / "target_encoder")
  logging.info("loading latent diffusion model")
  ddm_model, full_cfg, ddm_ema_params = load_ddm(
    weights_dir / "latent_diffusion"
  )

  @jax.jit
  def context_enc_fn(rngs, inputs):
    z, _ = ctx_model.apply(
      {"params": ctx_params},
      rngs=rngs,
      inputs=inputs,
      is_training=False,
      method=ctx_model.encode,
    )
    return z

  @jax.jit
  def target_dec_fn(_, inputs):
    return tgt_model.apply(
      {"params": tgt_params},
      inputs=inputs,
      is_training=False,
      method=tgt_model.decode,
    )

  repr_cfg = full_cfg.representation
  repr_fn, inv_repr_fn = processing_fns(
    variable_names="context",
    repres=repr_cfg,
    max_len=repr_cfg.max_len,
  )

  ddm_sample_fn = _make_sample_fn(
    full_cfg.model.name,
    ddm_model.apply,
    full_cfg.model.sampler.n_steps,
    full_cfg.training.ema_rate,
  )

  def _sample(rng_key, state, batch):
    ctx_key, sample_key = jr.split(rng_key)
    z_context = context_enc_fn({"sample": ctx_key}, batch["context"])
    z_target = ddm_sample_fn(
      sample_key, state=state, context=z_context, condition=batch["condition"]
    )
    return target_dec_fn(None, z_target)

  return GenerationPipeline(
    p_sample_fn=jax.pmap(_sample, axis_name="batch"),
    pstate=jax_utils.replicate(_InferenceState(ema_params=ddm_ema_params)),
    repr_fn=repr_fn,
    inv_repr_fn=inv_repr_fn,
    max_len=repr_cfg.max_len,
    condition_keys=_CONDITION_KEYS,
  )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

# Order of the metadata values the DiT was conditioned on; matches the
# ``meta`` vector of the Amatrice dataset builder used for training.
_CONDITION_KEYS = (
  "magnitude",
  "vs30",
  "station_longitude",
  "station_latitude",
  "station_elevation",
  "hypocentre_depth",
  "hypocentre_longitude",
  "hypocentre_latitude",
)


def _check_input(path: str, condition_keys: tuple, min_len: int) -> None:
  """Checks that an input HDF5 file can be sampled from.

  Args:
    path: Input HDF5 file path.
    condition_keys: Metadata dataset names the file must contain.
    min_len: Minimum context length in samples.

  Raises:
    ValueError: If datasets are missing, ``context`` is not
      ``(N, L, 3)`` or ``(N, 3, L)``, or ``L < min_len``.
  """
  with h5py.File(path, "r") as f:
    missing = [k for k in ("context", *condition_keys) if k not in f]
    if missing:
      raise ValueError(f"{path} is missing datasets: {missing}")
    shape = f["context"].shape
  if len(shape) != 3 or 3 not in (shape[1], shape[2]):
    raise ValueError(
      f"context must have shape (N, L, 3) or (N, 3, L), got {shape}"
    )
  length = shape[1] if shape[-1] == 3 else shape[-1]
  if length < min_len:
    raise ValueError(
      f"context waveforms have {length} samples; the model needs at "
      f"least {min_len}"
    )


def _read_input_batches(
  path: str, entry_batch_size: int, condition_keys: tuple
):
  """Yield (start, total, context, meta) chunks from an input HDF5 file.

  Args:
    path: Input HDF5 file path.
    entry_batch_size: Number of input entries per chunk.
    condition_keys: Ordered metadata field names to read from the file.

  Yields:
    ``(start_idx, total_entries, context, meta)`` where ``context`` is
    float32 ``(B, L, 3)`` and ``meta`` is float32 ``(B, len(condition_keys))``.
  """
  with h5py.File(path, "r") as f:
    total = len(f["context"])
    for start in range(0, total, entry_batch_size):
      sl = slice(start, start + entry_batch_size)
      ctx = np.array(f["context"][sl], dtype=np.float32)
      if ctx.ndim == 3 and ctx.shape[-1] != 3:
        ctx = rearrange(ctx, "b c l -> b l c")
      meta = np.stack(
        [np.array(f[k][sl], dtype=np.float32) for k in condition_keys], axis=1
      )
      yield start, total, ctx, meta


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


def generate(
  pipeline: GenerationPipeline,
  input_path: str,
  output_path: str,
  n_per_entry: int = 1,
  batch_size_per_device: int = 64,
  seed: int = 0,
) -> None:
  """Generate synthetic target waveforms and write them to an HDF5 file.

  Args:
    pipeline: Loaded pipeline from :func:`build_pipeline`.
    input_path: Path to the input HDF5 file.
    output_path: Path for the output HDF5 file (overwritten if it exists).
    n_per_entry: Number of independent samples per input entry.
    batch_size_per_device: Per-device batch size for pmap.
    seed: RNG seed for reproducibility.
  """
  _check_input(input_path, pipeline.condition_keys, pipeline.max_len)
  rng = jr.PRNGKey(seed)
  n_dev = jax.device_count()
  entry_chunk = max(1, batch_size_per_device * n_dev // n_per_entry)

  with h5py.File(input_path, "r") as f_in:
    total_entries = len(f_in["context"])
  total_samples = total_entries * n_per_entry

  logging.info(
    "generating %d waveforms (%d entries × %d per entry) on %d device(s)",
    total_samples,
    total_entries,
    n_per_entry,
    n_dev,
  )

  with h5py.File(output_path, "w") as f_out:
    ds_hat = f_out.create_dataset(
      "target_hat", (total_samples, pipeline.max_len, 3), dtype="f4"
    )
    ds_ctx = f_out.create_dataset(
      "context", (total_samples, pipeline.max_len, 3), dtype="f4"
    )
    ds_meta = f_out.create_dataset(
      "meta", (total_samples, len(pipeline.condition_keys)), dtype="f4"
    )

    cursor = 0
    for start, total, ctx_raw, meta in _read_input_batches(
      input_path, entry_chunk, pipeline.condition_keys
    ):
      logging.info("  processing entries %d/%d", start, total)

      ctx_spec = np.array(
        pipeline.repr_fn({"context": ctx_raw, "meta": meta})["inputs"]
      )
      ctx_spec_rep = np.repeat(ctx_spec, n_per_entry, axis=0)
      meta_rep = np.repeat(meta, n_per_entry, axis=0)
      ctx_raw_rep = np.repeat(
        ctx_raw[:, : pipeline.max_len, :], n_per_entry, axis=0
      )

      n_batch = ctx_spec_rep.shape[0]
      pad = (-n_batch) % n_dev
      if pad:
        ctx_spec_pad = np.concatenate([ctx_spec_rep, ctx_spec_rep[:pad]])
        meta_pad = np.concatenate([meta_rep, meta_rep[:pad]])
      else:
        ctx_spec_pad, meta_pad = ctx_spec_rep, meta_rep

      pbatch = common_utils.shard(
        {"context": jnp.array(ctx_spec_pad), "condition": jnp.array(meta_pad)}
      )
      rng, sample_rng = jr.split(rng)
      specs_hat = pipeline.p_sample_fn(
        jr.split(sample_rng, n_dev), pipeline.pstate, pbatch
      )
      specs_hat = np.array(specs_hat.reshape(-1, *specs_hat.shape[2:]))[
        :n_batch
      ]

      signals_hat = np.array(pipeline.inv_repr_fn(jnp.array(specs_hat)))
      min_len = min(
        signals_hat.shape[1], pipeline.max_len, ctx_raw_rep.shape[1]
      )

      end = cursor + n_batch
      ds_hat[cursor:end, :min_len] = signals_hat[:, :min_len]
      ds_ctx[cursor:end, :min_len] = ctx_raw_rep[:, :min_len]
      ds_meta[cursor:end] = meta_rep
      cursor = end

  logging.info("wrote %d samples to %r", cursor, output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
  """Command-line entry point for ``generate-waveforms``."""
  parser = argparse.ArgumentParser(
    prog="generate-waveforms",
    description=_DESCRIPTION,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "--input",
    required=True,
    help="Input HDF5 file with context waveforms and metadata.",
  )
  parser.add_argument(
    "--output",
    required=True,
    help="Output HDF5 file for generated waveforms.",
  )
  parser.add_argument(
    "--n-per-entry",
    type=int,
    default=1,
    dest="n_per_entry",
    help="Waveforms to generate per input entry (default: 1).",
  )
  parser.add_argument(
    "--batch-size-per-device",
    type=int,
    default=64,
    dest="batch_size_per_device",
    help="Per-device batch size (default: 64).",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="RNG seed (default: 0).",
  )
  parser.add_argument(
    "--weights-dir",
    default=None,
    dest="weights_dir",
    help="Override the weights cache directory (skips auto-download).",
  )
  args = parser.parse_args()
  logging.set_verbosity(logging.INFO)

  weights_dir = Path(args.weights_dir) if args.weights_dir else get_weights()
  pipeline = build_pipeline(weights_dir)
  generate(
    pipeline=pipeline,
    input_path=args.input,
    output_path=args.output,
    n_per_entry=args.n_per_entry,
    batch_size_per_device=args.batch_size_per_device,
    seed=args.seed,
  )


if __name__ == "__main__":
  main()
