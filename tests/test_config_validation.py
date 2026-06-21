"""__post_init__ validation for every tqe config dataclass."""

import pytest

from tqe.autoencoder import AutoencoderConfig
from tqe.denoising_diffusion import EDMParameterization
from tqe.diffusion_consistency_distillation import (
  EDMConsistencyDistillationConfig,
)
from tqe.flow_matching import FlowMatchingConfig
from tqe.wgan import WGANConfig

# ---------------------------------------------------------------------------
# WGANConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    ({"lamb": 0.0}, "lamb must be > 0"),
    ({"lamb": -5.0}, "lamb must be > 0"),
    ({"n_critic_steps": 0}, "n_critic_steps must be >= 1"),
    ({"n_critic_steps": -1}, "n_critic_steps must be >= 1"),
  ],
)
def test_wgan_config_validation(kwargs, match):
  with pytest.raises(ValueError, match=match):
    WGANConfig(**kwargs)


def test_wgan_config_valid():
  cfg = WGANConfig(lamb=5.0, n_critic_steps=3)
  assert cfg.lamb == 5.0
  assert cfg.n_critic_steps == 3


# ---------------------------------------------------------------------------
# FlowMatchingConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    ({"n_sampling_steps": 0}, "n_sampling_steps must be >= 1"),
    ({"n_sampling_steps": -1}, "n_sampling_steps must be >= 1"),
    (
      {"n_sampling_steps": 1, "time_eps": 1.0, "time_max": 0.5},
      "time_eps",
    ),
    (
      {"n_sampling_steps": 1, "time_eps": -0.1},
      "time_eps",
    ),
    (
      {"n_sampling_steps": 1, "time_embedding_scale": 0.0},
      "time_embedding_scale must be > 0",
    ),
    (
      {"n_sampling_steps": 1, "time_embedding_scale": -1.0},
      "time_embedding_scale must be > 0",
    ),
  ],
)
def test_flow_matching_config_validation(kwargs, match):
  with pytest.raises(ValueError, match=match):
    FlowMatchingConfig(**kwargs)


def test_flow_matching_config_valid():
  cfg = FlowMatchingConfig(n_sampling_steps=10)
  assert cfg.n_sampling_steps == 10
  assert cfg.time_embedding_scale == 999.0


# ---------------------------------------------------------------------------
# EDMParameterization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    ({"n_sampling_steps": 1}, "n_sampling_steps must be >= 2"),
    ({"n_sampling_steps": 2, "sigma_min": 0.0}, "sigma_min must be > 0"),
    ({"n_sampling_steps": 2, "sigma_min": -1.0}, "sigma_min must be > 0"),
    (
      {"n_sampling_steps": 2, "sigma_min": 100.0, "sigma_max": 80.0},
      "sigma_min.*must be < sigma_max",
    ),
    ({"n_sampling_steps": 2, "rho": 0.0}, "rho must be > 0"),
    ({"n_sampling_steps": 2, "sigma_data": 0.0}, "sigma_data must be > 0"),
    ({"n_sampling_steps": 2, "P_std": 0.0}, "P_std must be > 0"),
  ],
)
def test_edm_config_validation(kwargs, match):
  with pytest.raises(ValueError, match=match):
    EDMParameterization(**kwargs)


def test_edm_config_valid():
  cfg = EDMParameterization(n_sampling_steps=5)
  assert cfg.n_sampling_steps == 5


# ---------------------------------------------------------------------------
# AutoencoderConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    ({"kl_weight": -1e-6}, "kl_weight must be >= 0"),
    ({"kl_weight": -1.0}, "kl_weight must be >= 0"),
  ],
)
def test_autoencoder_config_validation(kwargs, match):
  with pytest.raises(ValueError, match=match):
    AutoencoderConfig(**kwargs)


def test_autoencoder_config_zero_kl_valid():
  cfg = AutoencoderConfig(kl_weight=0.0)
  assert cfg.kl_weight == 0.0


# ---------------------------------------------------------------------------
# EDMConsistencyDistillationConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    (
      {"n_sampling_steps": 0, "consistency_condition": "edm_style"},
      "n_sampling_steps must be >= 1",
    ),
    (
      {"n_sampling_steps": 2, "consistency_condition": "unknown"},
      "unknown consistency_condition",
    ),
    (
      {
        "n_sampling_steps": 2,
        "consistency_condition": "edm_style",
        "sigma_min": 0.0,
      },
      "sigma_min must be > 0",
    ),
    (
      {
        "n_sampling_steps": 2,
        "consistency_condition": "edm_style",
        "sigma_min": 100.0,
        "sigma_max": 80.0,
      },
      "sigma_min.*must be < sigma_max",
    ),
    (
      {"n_sampling_steps": 2, "consistency_condition": "edm_style", "rho": 0.0},
      "rho must be > 0",
    ),
    (
      {
        "n_sampling_steps": 2,
        "consistency_condition": "edm_style",
        "sigma_data": 0.0,
      },
      "sigma_data must be > 0",
    ),
    (
      {
        "n_sampling_steps": 2,
        "consistency_condition": "edm_style",
        "P_std": 0.0,
      },
      "P_std must be > 0",
    ),
  ],
)
def test_cd_config_validation(kwargs, match):
  with pytest.raises(ValueError, match=match):
    EDMConsistencyDistillationConfig(**kwargs)


def test_cd_config_valid():
  cfg = EDMConsistencyDistillationConfig(2, "edm_style")
  assert cfg.n_sampling_steps == 2
  assert callable(cfg.consistency_function)


def test_cd_config_no_var_data_attribute():
  cfg = EDMConsistencyDistillationConfig(2, "edm_style")
  assert not hasattr(cfg, "var_data"), (
    "var_data should not leak as an attribute"
  )
