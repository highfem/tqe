"""Shared return types for tqe objective factories."""

from collections.abc import Callable
from typing import NamedTuple


class TrainFns(NamedTuple):
  """Paired step and eval functions from an objective factory.

  Both share the seam ``(rng_key, state, batch) -> (metrics, new_state)``.
  """

  step_fn: Callable
  eval_fn: Callable


class ObjectiveFns(NamedTuple):
  """Return type of every tqe objective factory.

  Positional unpacking is preserved for backward compatibility::

    (step_fn, eval_fn), sample_fn, extra = factory(...)
    # or, named:
    obj = factory(...)
    obj.train_fns.step_fn(rng_key, state, batch)
  """

  train_fns: TrainFns
  sample_fn: Callable | None
  extra: Callable | None
