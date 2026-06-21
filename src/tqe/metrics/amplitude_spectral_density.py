"""Amplitude spectral density evaluation callback."""

import dataclasses

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from tqe.metrics.callback import Callback
from tqe.metrics.summarize import compute_running_mean_and_var


@dataclasses.dataclass(frozen=True)
class AmplitudeSpectralDensityConfig:
  """Configuration for amplitude spectral density evaluation.

  Attributes:
    fs: Sampling frequency in Hz.
    channel: Waveform channel index to evaluate.
    log_eps: Floor value for log-amplitude clipping.
  """

  fs: float
  channel: int = 0
  log_eps: float = 1e-8


def log_amplitude_spectral_density(
  signal: np.ndarray, log_eps: float
) -> np.ndarray:
  """Returns log-amplitude spectrum of ``signal`` clipped to ``[log_eps, ∞)``."""
  amplitude = np.abs(np.fft.rfft(signal, axis=-1))
  return np.log(np.clip(amplitude, log_eps, None))


def frechet_distance(
  mu_x: np.ndarray,
  std_x: np.ndarray,
  mu_y: np.ndarray,
  std_y: np.ndarray,
) -> float:
  """Fréchet distance between two diagonal Gaussians parameterised by mean and std.

  Args:
    mu_x: Mean of the first distribution, shape ``(n_freq,)``.
    std_x: Standard deviation of the first distribution, shape ``(n_freq,)``.
    mu_y: Mean of the second distribution, shape ``(n_freq,)``.
    std_y: Standard deviation of the second distribution, shape ``(n_freq,)``.

  Returns:
    Scalar Fréchet distance ``‖mu_x − mu_y‖² + ‖std_x − std_y‖²``.
  """
  return float(np.sum((mu_x - mu_y) ** 2) + np.sum((std_x - std_y) ** 2))


def _align_channel(
  target: np.ndarray, pred: np.ndarray, channel: int
) -> tuple[np.ndarray, np.ndarray, int]:
  n_time = min(target.shape[1], pred.shape[1])
  return (
    target[:, :n_time, channel],
    pred[:, :n_time, channel],
    n_time,
  )


def _per_batch_summary(
  target: np.ndarray, pred: np.ndarray, channel: int, log_eps: float
) -> tuple[dict, dict]:
  """Computes per-batch log-ASD statistics for one channel."""
  target, pred, n_time = _align_channel(target, pred, channel)
  target_sd = log_amplitude_spectral_density(target, log_eps)
  pred_sd = log_amplitude_spectral_density(pred, log_eps)
  target_stats = {
    "batch_size": np.array([target_sd.shape[0]]),
    "mean": target_sd.mean(axis=0),
    "var": target_sd.var(axis=0),
    "n_time": n_time,
  }
  pred_stats = {
    "batch_size": np.array([pred_sd.shape[0]]),
    "mean": pred_sd.mean(axis=0),
    "var": pred_sd.var(axis=0),
    "n_time": n_time,
  }
  return target_stats, pred_stats


def _mean_and_std(batch_summaries: list) -> tuple[np.ndarray, np.ndarray]:
  """Aggregates a list of per-batch summaries into a global mean and std."""
  # compute_running_mean_and_var requires consistent carry keys; strip n_time
  # which is scalar metadata carried alongside the spectral stats.
  stats = [
    {"batch_size": s["batch_size"], "mean": s["mean"], "var": s["var"]}
    for s in batch_summaries
  ]
  merged = compute_running_mean_and_var(stats)
  return merged["mean"], np.sqrt(merged["var"])


def _plot_spectral_distributions(
  target_mean: np.ndarray,
  target_std: np.ndarray,
  pred_mean: np.ndarray,
  pred_std: np.ndarray,
  fs: float,
  n_time: int,
) -> plt.Figure:
  freq = np.fft.rfftfreq(n_time, d=1 / fs)
  log_freq = np.log(np.where(freq > 0, freq, np.nan))  # DC bin → NaN, not -inf

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.plot(log_freq, pred_mean, color="b", label="Predicted")
  ax.fill_between(
    log_freq,
    pred_mean - pred_std,
    pred_mean + pred_std,
    color="b",
    alpha=0.2,
  )
  ax.plot(log_freq, target_mean, color="orange", label="Target")
  ax.fill_between(
    log_freq,
    target_mean - target_std,
    target_mean + target_std,
    color="orange",
    alpha=0.2,
  )
  ax.set_xlabel("Log-Frequency [Hz]")
  ax.set_ylabel(r"Log-Amplitude $[m/s^2 \ Hz^{-1}]$")
  ax.legend()
  fig.tight_layout()
  return fig


def amplitude_spectral_density(
  config: AmplitudeSpectralDensityConfig,
) -> Callback:
  """Builds a ``Callback`` that compares log-amplitude spectral densities.

  Args:
    config: Sampling rate, channel, and log-clipping settings.

  Returns:
    Callback that summarizes, aggregates, and optionally plots spectra.
  """
  name = f"amplitude spectral density, channel-{config.channel}"

  def summarize(target: np.ndarray, pred: np.ndarray):
    return _per_batch_summary(target, pred, config.channel, config.log_eps)

  def compute(targets: list, preds: list):
    target_mean, target_std = _mean_and_std(targets)
    pred_mean, pred_std = _mean_and_std(preds)
    distance = frechet_distance(pred_mean, pred_std, target_mean, target_std)
    return jnp.array([distance])

  def plot(targets: list, preds: list):
    n_time = targets[0]["n_time"]
    target_mean, target_std = _mean_and_std(targets)
    pred_mean, pred_std = _mean_and_std(preds)
    return _plot_spectral_distributions(
      target_mean, target_std, pred_mean, pred_std, config.fs, n_time
    )

  return Callback(name=name, summarize=summarize, compute=compute, plot=plot)
