"""Evaluation callback protocol for generative metrics."""

from collections.abc import Callable
from typing import Any, NamedTuple


class Callback(NamedTuple):
  """Evaluation callback with summarize, compute, and optional plot hooks.

  Attributes:
    name: Human-readable identifier used as a metrics key.
    summarize: Maps a target/pred batch pair to per-batch summary dicts.
    compute: Aggregates lists of target and pred summaries into a metric.
    plot: Optional function that renders a figure from aggregated summaries.
  """

  name: str
  summarize: Callable[[Any, Any], tuple[dict, dict]]
  compute: Callable[[list, list], Any]
  plot: Callable[[list, list], Any] | None = None
