import pathlib


def test_root_exports_objectives():
  import tqe  # noqa: PLC0415

  assert callable(tqe.flow_matching)
  assert callable(tqe.autoencoder)


def test_domain_namespace():
  from tqe.nn import dit  # noqa: PLC0415

  assert hasattr(dit, "make_model")
  assert not hasattr(dit, "load_model_with_ema_weights")
  assert not hasattr(dit, "construct_apply_fn")


def test_tqe_has_no_experiment_imports():
  root = pathlib.Path(__file__).resolve().parents[1] / "src" / "tqe"
  offenders = []
  for path in sorted(root.rglob("*.py")):
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
      stripped = line.strip()
      if stripped.startswith(("import experiments", "from experiments")):
        offenders.append(f"{path}:{lineno}: {stripped}")
  assert not offenders, "tqe must not import experiments:\n" + "\n".join(
    offenders
  )
