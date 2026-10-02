import pickle
import tarfile

import package_weights
import pytest

_TRAINING_HASHES = {
  "context_encoder": (
    "2564da3fb2d54a2bf35b78a86173a52792dbabfeaa5b69e24c93de66282b3ff0"
  ),
  "target_encoder": (
    "4b96590036e42c196e88060ca47f1838b2c0935c1e05324380ba1c05c0e04307"
  ),
  "latent_diffusion": (
    "5a215ab301fc963480b50448ec71eb3f1ba765eb6cc6d21215ffc5ad76d071a6"
  ),
}


@pytest.mark.parametrize("name", sorted(_TRAINING_HASHES))
def test_configs_reproduce_training_run_hash(name):
  config = package_weights.training_config(
    package_weights._ARCHIVES[name], name
  )
  assert package_weights.run_hash(config) == _TRAINING_HASHES[name]


def test_write_archive_ships_best_and_full_config_only(tmp_path):
  checkpoint_dir = tmp_path / "latent_diffusion-abc"
  (checkpoint_dir / "best").mkdir(parents=True)
  (checkpoint_dir / "last" / "100").mkdir(parents=True)
  (checkpoint_dir / "config.pkl").write_bytes(pickle.dumps({"name": "fm"}))
  (checkpoint_dir / "best" / "_METADATA").write_text("{}")
  (checkpoint_dir / "last" / "100" / "_METADATA").write_text("{}")

  archive = package_weights.write_archive(
    checkpoint_dir, "latent_diffusion", tmp_path, {"model": {}}
  )

  with tarfile.open(archive) as tar:
    names = set(tar.getnames())
    full_config = pickle.load(
      tar.extractfile("latent_diffusion/full_config.pkl")
    )
  assert names == {
    "latent_diffusion/config.pkl",
    "latent_diffusion/best",
    "latent_diffusion/best/_METADATA",
    "latent_diffusion/full_config.pkl",
  }
  assert full_config == {"model": {}}
