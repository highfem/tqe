"""Tests for tqe.metrics: frechet_distance, running aggregation, and ASD callback."""

import numpy as np
import pytest

from tqe.metrics.amplitude_spectral_density import (
  AmplitudeSpectralDensityConfig,
  amplitude_spectral_density,
  frechet_distance,
  log_amplitude_spectral_density,
)
from tqe.metrics.summarize import compute_running_mean_and_var

# ---------------------------------------------------------------------------
# log_amplitude_spectral_density
# ---------------------------------------------------------------------------


def test_log_asd_shape():
  signal = np.random.default_rng(0).normal(size=(4, 100))
  result = log_amplitude_spectral_density(signal, log_eps=1e-8)
  assert result.shape == (4, 51)  # rfft of length-100 → 51 freq bins


def test_log_asd_floor():
  signal = np.zeros((2, 64))
  result = log_amplitude_spectral_density(signal, log_eps=1e-8)
  assert np.allclose(result, np.log(1e-8))


# ---------------------------------------------------------------------------
# frechet_distance
# ---------------------------------------------------------------------------


def test_frechet_distance_identical_is_zero():
  mu = np.array([1.0, 2.0, 3.0])
  std = np.array([0.5, 1.0, 1.5])
  assert frechet_distance(mu, std, mu, std) == 0.0


def test_frechet_distance_mean_shift():
  mu_x = np.array([0.0, 0.0])
  mu_y = np.array([1.0, 0.0])
  std = np.ones(2)
  dist = frechet_distance(mu_x, std, mu_y, std)
  assert dist == pytest.approx(1.0)


def test_frechet_distance_std_shift():
  mu = np.zeros(2)
  std_x = np.array([1.0, 1.0])
  std_y = np.array([2.0, 1.0])
  dist = frechet_distance(mu, std_x, mu, std_y)
  assert dist == pytest.approx(1.0)


def test_frechet_distance_positive():
  rng = np.random.default_rng(42)
  mu_x, mu_y = rng.normal(size=(2, 8))
  std_x, std_y = np.abs(rng.normal(size=(2, 8)))
  dist = frechet_distance(mu_x, std_x, mu_y, std_y)
  assert dist >= 0.0
  assert isinstance(dist, float)


# ---------------------------------------------------------------------------
# compute_running_mean_and_var
# ---------------------------------------------------------------------------


def _make_summary(batch_size, mean_val, var_val, n_freq=4):
  return {
    "batch_size": np.array([batch_size]),
    "mean": np.full(n_freq, mean_val),
    "var": np.full(n_freq, var_val),
  }


def test_running_mean_two_equal_batches():
  summaries = [_make_summary(2, 2.0, 0.5), _make_summary(2, 4.0, 0.5)]
  result = compute_running_mean_and_var(summaries)
  np.testing.assert_allclose(result["mean"], 3.0, atol=1e-5)
  np.testing.assert_allclose(result["var"], 1.5, atol=1e-5)


def test_running_mean_single_batch_is_identity():
  s = _make_summary(3, 7.0, 2.0)
  result = compute_running_mean_and_var([s, _make_summary(0, 0.0, 0.0)])
  np.testing.assert_allclose(result["mean"], 7.0, atol=1e-5)


def test_running_mean_three_equal_batches():
  """Three identical batches should return the same mean and var."""
  s = _make_summary(4, 5.0, 1.0)
  result = compute_running_mean_and_var([s, s, s])
  np.testing.assert_allclose(result["mean"], 5.0, atol=1e-5)
  np.testing.assert_allclose(result["var"], 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# amplitude_spectral_density callback (end-to-end)
# ---------------------------------------------------------------------------


def _make_signals(n_batch=8, n_time=256, n_channels=3, seed=0):
  return np.random.default_rng(seed).normal(size=(n_batch, n_time, n_channels))


def test_asd_summarize_returns_dict_pair():
  cfg = AmplitudeSpectralDensityConfig(fs=100.0, channel=0)
  cb = amplitude_spectral_density(cfg)
  target = _make_signals()
  pred = _make_signals(seed=1)
  t_stats, p_stats = cb.summarize(target, pred)
  for stats in (t_stats, p_stats):
    assert "batch_size" in stats
    assert "mean" in stats
    assert "var" in stats
    assert stats["mean"].shape == (129,)  # rfft of 256 → 129 bins


def test_asd_compute_returns_scalar():
  cfg = AmplitudeSpectralDensityConfig(fs=100.0, channel=0)
  cb = amplitude_spectral_density(cfg)
  target = _make_signals()
  pred = _make_signals(seed=1)
  t_stats, p_stats = cb.summarize(target, pred)
  metric = cb.compute([t_stats, t_stats], [p_stats, p_stats])
  assert metric.shape == (1,)
  assert float(metric[0]) >= 0.0


def test_asd_identical_signals_near_zero_distance():
  cfg = AmplitudeSpectralDensityConfig(fs=100.0, channel=0)
  cb = amplitude_spectral_density(cfg)
  target = _make_signals()
  t_stats, p_stats = cb.summarize(target, target)
  metric = cb.compute([t_stats], [p_stats])
  assert float(metric[0]) == pytest.approx(0.0, abs=1e-6)


def test_asd_callback_name():
  cfg = AmplitudeSpectralDensityConfig(fs=100.0, channel=2)
  cb = amplitude_spectral_density(cfg)
  assert "channel-2" in cb.name


def test_asd_plot_returns_figure():
  pytest.importorskip("matplotlib")
  cfg = AmplitudeSpectralDensityConfig(fs=100.0, channel=0)
  cb = amplitude_spectral_density(cfg)
  target = _make_signals(n_batch=4, n_time=64)
  pred = _make_signals(n_batch=4, n_time=64, seed=2)
  t_stats, p_stats = cb.summarize(target, pred)
  fig = cb.plot([t_stats], [p_stats])
  assert fig is not None
