from typing import Any

import numpy as np
import tensorflow_datasets as tfds
from h5py import File


class AmatricePairedWaveforms(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")
  RELEASE_NOTES = {
    "1.0.0": "Initial release.",
  }

  def __init__(self, **kwargs: Any):
    super().__init__(**kwargs)

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    adata = self.dataset_info_from_configs(
      features=tfds.features.FeaturesDict(
        {
          "target": tfds.features.Tensor(shape=(None, 3), dtype=np.float32),
          "context": tfds.features.Tensor(shape=(None, 3), dtype=np.float32),
          "meta": tfds.features.Tensor(shape=(None,), dtype=np.float32),
        }
      ),
    )
    return adata

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    path = "/capstor/scratch/cscs/sdirmeie/PROJECTS/highfem/workdir/data-raw/raw_waveforms_amatrice.h5"
    waveforms = File(path, "r")

    rng = np.random.default_rng(seed=42)
    indices = np.arange(len(waveforms["waveforms"][:]))
    indices = rng.permutation(indices)

    return {
      "train": self._generate_examples(waveforms, indices, is_training=True),
      "test": self._generate_examples(waveforms, indices, is_training=False),
    }

  def _generate_examples(self, waveforms, indices, is_training):
    cutoff = 9000
    num_train_samples = int(len(indices) * 0.9)
    if is_training:
      indices = indices[:num_train_samples]
    else:
      indices = indices[num_train_samples:]

    for ii in range(len(indices)):
      idx = int(indices[ii])
      target, context = waveforms["waveforms"][idx], waveforms["context"][idx]

      if target.shape != context.shape:
        min_shape = np.minimum(target.shape[-1], context.shape[-1])
        target, context = target[:, :min_shape], context[:, :min_shape]
      if target.shape[1] > cutoff:
        target, context = target[:, :cutoff], context[:, :cutoff]
      if target.shape != (3, cutoff):
        continue

      sum_zeros_target = np.array([np.sum(target[i] == 0.0) for i in range(3)])
      var_target = np.array([np.var(target[i]) for i in range(3)])
      if np.any((sum_zeros_target == cutoff) & (var_target == 0.0)):
        continue

      meta = np.array(
        [
          float(waveforms["magnitude"][idx]),
          float(waveforms["vs30"][idx]),
          float(waveforms["station_longitude"][idx]),
          float(waveforms["station_latitude"][idx]),
          float(waveforms["station_elevation"][idx]),
          float(waveforms["hypocentre_depth"][idx]),
          float(waveforms["hypocentre_longitude"][idx]),
          float(waveforms["hypocentre_latitude"][idx]),
        ]
      )

      yield (
        ii,
        {
          "target": target.T.astype(np.float32),
          "context": context.T.astype(np.float32),
          "meta": meta.astype(np.float32),
        },
      )
