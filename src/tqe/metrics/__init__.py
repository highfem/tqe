"""Training and evaluation metric callbacks."""

from tqe.metrics.amplitude_spectral_density import (
  AmplitudeSpectralDensityConfig,
  amplitude_spectral_density,
)
from tqe.metrics.callback import Callback

_CALLBACKS = {
  "amplitude_spectral_density": (
    amplitude_spectral_density,
    AmplitudeSpectralDensityConfig,
  ),
}


def build_callbacks(callbacks):
  """Instantiate callbacks from a sequence of ``{name, params}`` dicts."""
  result = []
  for cb in callbacks:
    name = cb["name"]
    if name not in _CALLBACKS:
      raise ValueError(
        f"unknown callback '{name}'; available: {sorted(_CALLBACKS)}"
      )
    factory, cfg_cls = _CALLBACKS[name]
    result.append(factory(cfg_cls(**cb["params"])))
  return result


__all__ = [
  "amplitude_spectral_density",
  "AmplitudeSpectralDensityConfig",
  "build_callbacks",
  "Callback",
]
