"""Package pretrained checkpoints into the archives downloaded by tqe.

Writes context_encoder.tar.gz, target_encoder.tar.gz and
latent_diffusion.tar.gz. Each holds <name>/config.pkl and <name>/best/;
the latent diffusion archive also holds <name>/full_config.pkl, the
training config that generate_waveforms reads the representation from.

Checkpoint folders are looked up by the run hash of the config files in
configs/, so a config that differs from the training config fails here.

Usage:
  python package_weights.py --weights-dir=<DIR> --outdir=<DIR>
"""

import argparse
import hashlib
import importlib.util
import io
import pathlib
import pickle
import tarfile
import time

_CONFIGS_DIR = pathlib.Path(__file__).resolve().parent / "configs"
_ARCHIVES = {
  "context_encoder": "config-autoencoder.py",
  "target_encoder": "config-autoencoder.py",
  "latent_diffusion": "config-ddm-fm-dit.py",
}


def training_config(config_file, model_type):
  spec = importlib.util.spec_from_file_location(
    "config", _CONFIGS_DIR / config_file
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  config = module.get_config()
  config.model_type = model_type
  config = config.to_dict()
  # the callbacks block was added to the config files after training
  # (d6d6e85, 9ff9a16) and is not part of the training run hash
  del config["callbacks"]
  return config


def run_hash(config):
  # same as main.hash_value; main.py is not imported because it pulls in
  # wandb and tensorflow
  return hashlib.sha256(str(config).encode("utf-8")).hexdigest()


def write_archive(checkpoint_dir, name, outdir, full_config=None):
  archive = outdir / f"{name}.tar.gz"
  with tarfile.open(archive, "w:gz") as tar:
    tar.add(checkpoint_dir / "config.pkl", arcname=f"{name}/config.pkl")
    tar.add(checkpoint_dir / "best", arcname=f"{name}/best")
    if full_config is not None:
      data = pickle.dumps(full_config)
      info = tarfile.TarInfo(f"{name}/full_config.pkl")
      info.size = len(data)
      info.mtime = int(time.time())
      info.mode = 0o644
      tar.addfile(info, io.BytesIO(data))
  return archive


def sha256_file(path):
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def main():
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--weights-dir",
    required=True,
    type=pathlib.Path,
    help="folder with the <model_type>-<hash> checkpoint folders",
  )
  parser.add_argument(
    "--outdir",
    required=True,
    type=pathlib.Path,
    help="folder the .tar.gz archives are written to",
  )
  args = parser.parse_args()
  args.outdir.mkdir(parents=True, exist_ok=True)

  print("_WEIGHTS = {")
  for name, config_file in _ARCHIVES.items():
    config = training_config(config_file, name)
    checkpoint_dir = args.weights_dir / f"{name}-{run_hash(config)}"
    if not checkpoint_dir.is_dir():
      raise FileNotFoundError(
        f"{checkpoint_dir} not found: configs/{config_file} does not match "
        f"the training config of any {name} checkpoint"
      )
    full_config = config if name == "latent_diffusion" else None
    archive = write_archive(checkpoint_dir, name, args.outdir, full_config)
    print(f'  "{archive.name}": "sha256:{sha256_file(archive)}",')
  print("}")


if __name__ == "__main__":
  main()
