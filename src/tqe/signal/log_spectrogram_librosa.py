"""Librosa STFT log-spectrogram encode/decode."""

import numpy as np
from einops import rearrange
from librosa import griffinlim, stft


def log_spectrogram_fns(
  stft_channels, hop_size, max_len, clip=1e-8, log_max=3.0
):
  """Returns forward and inverse log-spectrogram transforms.

  Args:
    stft_channels: FFT size used by ``librosa.stft``.
    hop_size: STFT hop length in samples.
    max_len: Unused; kept for API compatibility with the TensorFlow variant.
    clip: Minimum magnitude before taking the logarithm.
    log_max: Upper log-magnitude bound used for normalization to ``[-1, 1]``.

  Returns:
    A ``(as_spectrogram_fn, as_waveform_fn)`` pair for encoding and decoding.
  """
  log_clip = np.log(clip)
  stft_fn = lambda x: stft(
    x, n_fft=stft_channels, hop_length=hop_size, dtype=np.float32
  )
  istft_fn = lambda x: griffinlim(
    x,
    hop_length=hop_size,
    n_fft=stft_channels,
    n_iter=128,
    random_state=0,
    dtype=np.float32,
  )

  def _post_process_and_scale(spec):
    spec = np.abs(spec)
    log_spec = np.log(np.clip(spec, clip, None))
    norm_log_spec = (log_spec - log_clip) / (log_max - log_clip)  # [0, 1]
    norm_log_spec = norm_log_spec * 2 - 1  # [-1, 1]
    return norm_log_spec

  def as_spectrogram_fn(waveform):
    shape = waveform.shape
    waveform = rearrange(waveform, "b l c -> (b c) l")
    spec = np.array([stft_fn(x) for x in waveform])
    spec = spec[:, :-1]  # remove nquist frequency
    spec = rearrange(
      spec, "(b c) h w -> b h w c", b=shape[0], c=shape[-1]
    )  # transpose and remove nquist frequency
    return _post_process_and_scale(spec)

  def _invert_post_process_and_scale(spec):
    norm_log_spec = (spec + 1) / 2  # [0, 1]
    log_spec = norm_log_spec * (log_max - log_clip) + log_clip
    spec = np.exp(log_spec)
    return spec

  def as_waveform_fn(spec):
    spec = _invert_post_process_and_scale(spec)
    shape = spec.shape
    spec = rearrange(spec, "b h w c -> (b c) h w")
    spec = np.concatenate([spec, np.zeros_like(spec[:, :1])], axis=1)
    waveform = np.array([istft_fn(x) for x in spec])
    waveform = rearrange(waveform, "(b c) l -> b l c", b=shape[0], c=shape[-1])
    return waveform

  return as_spectrogram_fn, as_waveform_fn
