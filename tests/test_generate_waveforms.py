"""Tests for the generate-waveforms sampling pipeline."""

import pickle

import h5py
import numpy as np
import orbax.checkpoint
import pytest
from jax import numpy as jnp
from jax import random as jr
from ml_collections import ConfigDict

from tqe import generate_waveforms as gw
from tqe.nn import dit, vae

_MAX_LEN = 8161
_SPEC_SHAPE = (128, 256, 3)
_LATENT_SHAPE = (32, 32, 2)
_VAE_NN = {
  "base_model": {
    "attention_resolutions": (),
    "channel_multipliers": (1, 2, 2, 4),
    "downsampling_strides": ((2, 2), (2, 2), (1, 2)),
    "dropout_rate": 0.0,
    "kernel_size": 3,
    "n_channels": 8,
    "n_groups": 2,
    "n_resnet_blocks": 1,
  },
  "encoder": {"n_out_channels": 2 * _LATENT_SHAPE[-1]},
  "decoder": {"n_out_channels": 3},
}
_DIT_MODEL = {
  "name": "flow_matching",
  "nn": {
    "name": "dit",
    "dit_score_net": {
      "do_context_conditioning": True,
      "do_context_projections": True,
      "do_meta_conditioning": True,
      "dropout_rate": 0.0,
      "n_blocks": 1,
      "n_channels": 32,
      "n_embedding_dimension": 16,
      "n_heads": 2,
      "n_out_channels": _LATENT_SHAPE[-1],
      "patch_size": 2,
    },
  },
  "sampler": {"n_steps": 2},
}
_FULL_CONFIG = {
  "model": _DIT_MODEL,
  "representation": {
    "representations": [
      ConfigDict(
        {
          "name": "log_spectrogram",
          "params": {
            "stft_channels": 256,
            "hop_size": 32,
            "max_len": _MAX_LEN,
          },
        }
      )
    ],
    "latent_channels": _LATENT_SHAPE[-1],
    "max_len": _MAX_LEN,
    "fs": 100,
  },
  "training": {"ema_rate": 0.9999},
}


def _apply(*args, **kwargs):
  del args, kwargs


def _write_pickle(path, obj):
  with open(path, "wb") as f:
    pickle.dump(obj, f)


def _save_best(checkpoint_dir, state):
  orbax.checkpoint.PyTreeCheckpointer().save(
    str(checkpoint_dir / "best"), {"state": state}
  )


def _init_vae():
  model = vae.make_model(ConfigDict(_VAE_NN))
  rngs = {"params": jr.PRNGKey(0), "sample": jr.PRNGKey(1)}
  inputs = jnp.zeros((1, *_SPEC_SHAPE))
  return model.init(rngs, inputs, is_training=False)["params"]


def _init_dit():
  model = dit.make_model(ConfigDict(_DIT_MODEL["nn"]["dit_score_net"]))
  latent = jnp.zeros((1, *_LATENT_SHAPE))
  return model.init(
    {"params": jr.PRNGKey(0), "dropout": jr.PRNGKey(1)},
    inputs=latent,
    times=jnp.zeros((1,)),
    context=latent,
    condition=jnp.zeros((1, 8)),
    is_training=False,
  )["params"]


@pytest.fixture(scope="module")
def weights_dir(tmp_path_factory):
  root = tmp_path_factory.mktemp("weights")
  for name in ("context_encoder", "target_encoder"):
    checkpoint_dir = root / name
    checkpoint_dir.mkdir()
    _write_pickle(checkpoint_dir / "config.pkl", {"nn": _VAE_NN})
    _save_best(checkpoint_dir, {"params": _init_vae()})
  checkpoint_dir = root / "latent_diffusion"
  checkpoint_dir.mkdir()
  _write_pickle(checkpoint_dir / "config.pkl", _DIT_MODEL)
  _write_pickle(checkpoint_dir / "full_config.pkl", _FULL_CONFIG)
  _save_best(checkpoint_dir, {"ema_params": _init_dit()})
  return root


def _write_input(path, n, length=9000, channel_first=False, drop_key=None):
  rng = np.random.default_rng(0)
  shape = (n, 3, length) if channel_first else (n, length, 3)
  keys = (
    "magnitude",
    "vs30",
    "station_longitude",
    "station_latitude",
    "station_elevation",
    "hypocentre_depth",
    "hypocentre_longitude",
    "hypocentre_latitude",
  )
  with h5py.File(path, "w") as f:
    f["context"] = rng.normal(scale=1e-3, size=shape).astype(np.float32)
    for i, key in enumerate(keys):
      if key != drop_key:
        f[key] = np.full(n, float(i), dtype=np.float32)


def test_make_sample_fn_selects_flow_matching():
  fn = gw._make_sample_fn("flow_matching", _apply, 2, 0.999)
  assert fn.__qualname__ == "flow_matching.<locals>.sample_fn"


def test_make_sample_fn_selects_edm():
  fn = gw._make_sample_fn("edm", _apply, 2, 0.999)
  assert fn.__qualname__ == "denoising_diffusion.<locals>.sample_fn"


def test_make_sample_fn_rejects_unknown_objective():
  with pytest.raises(ValueError, match="unknown objective 'wgan'"):
    gw._make_sample_fn("wgan", _apply, 2, 0.999)


def test_generate_writes_samples_for_each_entry(weights_dir, tmp_path):
  _write_input(tmp_path / "in.h5", n=3)
  pipeline = gw.build_pipeline(weights_dir)
  gw.generate(
    pipeline,
    str(tmp_path / "in.h5"),
    str(tmp_path / "out.h5"),
    n_per_entry=2,
    batch_size_per_device=4,
  )
  with h5py.File(tmp_path / "out.h5", "r") as f:
    assert f["target_hat"].shape == (6, _MAX_LEN, 3)
    assert f["context"].shape == (6, _MAX_LEN, 3)
    assert np.all(np.isfinite(f["target_hat"][:]))
    expected_meta = np.tile(np.arange(8, dtype=np.float32), (6, 1))
    np.testing.assert_array_equal(f["meta"][:], expected_meta)


def test_check_input_rejects_short_context(tmp_path):
  _write_input(tmp_path / "in.h5", n=1, length=6000)
  with pytest.raises(ValueError, match="6000 samples"):
    gw._check_input(str(tmp_path / "in.h5"), gw._CONDITION_KEYS, _MAX_LEN)


def test_check_input_names_missing_metadata(tmp_path):
  _write_input(tmp_path / "in.h5", n=1, drop_key="vs30")
  with pytest.raises(ValueError, match="vs30"):
    gw._check_input(str(tmp_path / "in.h5"), gw._CONDITION_KEYS, _MAX_LEN)


def test_check_input_accepts_channel_first(tmp_path):
  _write_input(tmp_path / "in.h5", n=1, channel_first=True)
  gw._check_input(str(tmp_path / "in.h5"), gw._CONDITION_KEYS, _MAX_LEN)
