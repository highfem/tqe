"""Running mean/variance aggregation for metric callbacks."""

import jax
from flax.training.common_utils import stack_forest


def compute_running_mean_and_var(batch_summaries: list[dict]) -> dict:
  """Aggregates per-batch statistics using Welford's online algorithm.

  Args:
    batch_summaries: Non-empty list of dicts, each with keys ``batch_size``
      (shape ``(1,)``), ``mean``, and ``var`` (both shape ``(n_freq,)``).

  Returns:
    A single dict with the same keys, aggregated across all batches.
  """

  def _welford_step(carry, scan):
    running_size, running_mean, running_var = (
      carry["batch_size"],
      carry["mean"],
      carry["var"],
    )
    size, mean, var = scan["batch_size"], scan["mean"], scan["var"]
    new_size = running_size + size
    new_mean = (running_size * running_mean + size * mean) / new_size
    lhs = (running_size / new_size) * running_var + (size / new_size) * var
    rhs = (running_size * size) / (new_size**2) * (running_mean - mean) ** 2
    new_var = lhs + rhs
    return {"batch_size": new_size, "mean": new_mean, "var": new_var}, None

  if len(batch_summaries) == 1:
    return batch_summaries[0]
  init = batch_summaries[0]
  scans = stack_forest(batch_summaries[1:])
  summary, _ = jax.lax.scan(_welford_step, init, scans)
  return summary
