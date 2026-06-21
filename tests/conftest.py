"""Shared pytest session setup: one JAX/TF init for the whole run."""

from __future__ import annotations

import os

# Must be set before the first jax import anywhere in the session.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.config.experimental.set_visible_devices([], "GPU")
jax.devices()
