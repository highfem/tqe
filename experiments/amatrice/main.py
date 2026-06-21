import hashlib
import importlib
import os
import pathlib

import jax
import tensorflow as tf
import wandb
from absl import app, flags, logging
from ml_collections import config_flags
from utils import log_jax_env

from tqe.metrics import build_callbacks

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "training configuration", lock_config=False
)
flags.DEFINE_string(
  "workdir", None, "out directory, i.e., place where results are written to"
)
flags.DEFINE_enum(
  "experiment",
  None,
  [
    "classifier",
    "autoencoder",
    "latent_diffusion",
    "latent_wgan",
    "latent_consistency_distillation",
  ],
  "experimental model to train",
)
flags.DEFINE_enum(
  "mode",
  "train",
  ["train", "evaluate"],
  "either train the (latent) model or simulate using it",
)
flags.DEFINE_bool("usewand", False, "register run to wandb")
flags.DEFINE_string("outfile", None, "h5 output path (evaluate mode)")
flags.DEFINE_integer(
  "batch-size-per-device", 64, "batch size PER DEVICE (GPU/CPU)"
)
flags.DEFINE_string(
  "model_checkpoint",
  "latent_diffusion",
  "checkpoint of the generative model to be evaluated",
)
flags.DEFINE_string(
  "classifier_checkpoint",
  "classifier",
  "checkpoint of the classifier used during evaluation",
)
flags.DEFINE_string(
  "ddm_checkpoint",
  None,
  "ddm checkpoint, eg, used for consistency distillation",
)
flags.DEFINE_string(
  "ddm_checkpoint_step",
  None,
  "latent ddm checkpoint step for latent consistency distillation",
)
flags.DEFINE_list(
  "autoencoder_checkpoints",
  ["target_encoder", "context_encoder"],
  "checkpoints used for models that need submodels, such as latent diffusion",
)
flags.DEFINE_list(
  "autoencoder_checkpoints_steps",
  ["best", "best"],
  "the training step of the checkpoint, or 'last' for the latest train step",
)
flags.mark_flags_as_required(["config", "workdir", "mode"])

EXPERIMENT_MODULES = {
  "classifier": "classifier_experiment",
  "autoencoder": "autoencoder_experiment",
  "latent_diffusion": "latent_diffusion_experiment",
  "latent_wgan": "latent_wgan_experiment",
  "latent_consistency_distillation": "latent_consistency_distillation_experiment",
}


def hash_value(config):
  h = hashlib.new("sha256")
  h.update(str(config).encode("utf-8"))
  return h.hexdigest()


def main(argv):
  tf.config.experimental.set_visible_devices([], "GPU")
  del argv

  logging.set_verbosity(logging.INFO)
  config = FLAGS.config.to_dict()
  hsh = hash_value(config)
  logging.info("run hash: %s", hsh)
  log_jax_env()
  run_name = f"{FLAGS.config.model_type}-{hsh}"

  workdir = FLAGS.workdir
  if not pathlib.Path(workdir).exists():
    pathlib.Path(workdir).mkdir(parents=True, exist_ok=False)

  if FLAGS.mode == "evaluate":
    if FLAGS.outfile is None:
      raise ValueError("--outfile is required for --mode=evaluate")
    if FLAGS.autoencoder_checkpoints is None:
      FLAGS.autoencoder_checkpoints = ["context_encoder", "target_encoder"]
    if FLAGS.usewand:
      wandb.init(
        project="highfem-tqe-evaluation",
        config=config,
        dir=os.path.join(FLAGS.workdir, "wandb"),
        settings=wandb.Settings(init_timeout=300),
      )
    from evaluation import evaluate  # noqa: PLC0415

    evaluate(FLAGS.config.rng_key, FLAGS)
    return

  if FLAGS.experiment is None:
    raise ValueError("--experiment is required for --mode=train")

  if FLAGS.usewand:
    wandb.init(
      project="highfem-tqe-training",
      config=config,
      dir=os.path.join(FLAGS.workdir, "wandb"),
      id=run_name,
      resume="allow",
      settings=wandb.Settings(init_timeout=300),
    )
    wandb.run.name = run_name

  callbacks = build_callbacks([cb.to_dict() for cb in FLAGS.config.callbacks])

  experiment = importlib.import_module(EXPERIMENT_MODULES[FLAGS.experiment])
  experiment.train(FLAGS.config.rng_key, FLAGS, callbacks, run_name)


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
