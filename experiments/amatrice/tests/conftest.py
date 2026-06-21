"""Shared pytest session setup: one JAX/TF init for the whole run."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Amatrice scripts use flat sibling imports (cwd = experiments/amatrice).
_AMATRICE_ROOT = Path(__file__).resolve().parents[1]
if str(_AMATRICE_ROOT) not in sys.path:
  sys.path.insert(0, str(_AMATRICE_ROOT))

# Must be set before the first jax import anywhere in the session.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.config.experimental.set_visible_devices([], "GPU")
jax.devices()
