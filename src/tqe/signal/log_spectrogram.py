"""TensorFlow STFT log-spectrogram encode/decode."""

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from einops import rearrange


def log_spectrogram_fns(
  stft_channels, hop_size, max_len, clip=1e-8, log_max=3.0
):
  """Returns forward and inverse log-spectrogram transforms.

  Args:
    stft_channels: FFT size and analysis window length in samples.
    hop_size: STFT hop length in samples.
    max_len: Maximum waveform length used for padding and trimming.
    clip: Minimum magnitude before taking the logarithm.
    log_max: Upper log-magnitude bound used for normalization to ``[-1, 1]``.

  Returns:
    A ``(as_spectrogram_fn, as_waveform_fn)`` pair for encoding and decoding.
  """
  log_clip = np.log(clip)
  stft = lambda x: tfio.audio.spectrogram(
    x, stft_channels, window=stft_channels, stride=hop_size
  )
  istft = lambda x: tfio.audio.inverse_spectrogram(
    x, stft_channels, stft_channels, hop_size
  )

  def _post_process_and_scale(spec):
    spec = tf.abs(spec)
    log_spec = tf.math.log(tf.clip_by_value(spec, clip, tf.float32.max))
    norm_log_spec = (log_spec - log_clip) / (log_max - log_clip)  # [0, 1]
    norm_log_spec = norm_log_spec * 2 - 1  # [-1, 1]
    return norm_log_spec

  @tf.numpy_function(Tout=tf.float32)
  def as_spectrogram_fn(waveform):
    waveform = tf.pad(
      waveform[:, : (max_len - stft_channels // 2), :],
      np.array([[0, 0], [stft_channels // 2, 0], [0, 0]]),
      mode="constant",
      constant_values=0,
    )
    shape = waveform.shape
    waveform = rearrange(waveform, "b l c -> (b c) l")
    spec = stft(waveform)
    spec = spec[..., :-1]  # remove Nyquist frequency
    spec = rearrange(
      spec, "(b c) w h -> b h w c", b=shape[0], c=shape[-1]
    )  # transpose and remove Nyquist frequency
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
    spec = rearrange(spec, "b h w -> b w h")
    waveform = istft(spec).numpy()
    waveform = rearrange(waveform, "(b c) l -> b l c", b=shape[0], c=shape[-1])
    waveform = waveform[:, (stft_channels // 2) : max_len, :]
    return waveform

  return as_spectrogram_fn, as_waveform_fn
