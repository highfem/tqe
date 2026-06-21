"""Butterworth low-pass filtering for waveform preprocessing."""

import chex
import numpy as np
import tensorflow as tf
from scipy.signal import iirfilter, sosfilt, zpk2sos


def lowpass_filter(freq, df, corners=4):
  fe = 0.5 * df
  f = freq / fe
  z, p, k = iirfilter(corners, f, btype="lowpass", ftype="butter", output="zpk")
  sos = zpk2sos(z, p, k)
  return sos


def taper(npts, p=0.1):
  if p == 0.0 or p == 1.0:
    frac = int(npts * p / 2.0)
  else:
    frac = int(npts * p / 2.0 + 0.5)

  idx1 = 0
  idx2 = frac - 1
  idx3 = npts - frac
  idx4 = npts - 1

  if idx1 == idx2:
    idx2 += 1
  if idx3 == idx4:
    idx3 -= 1

  cos_win = np.zeros(npts)
  cos_win[idx1 : idx2 + 1] = 0.5 * (
    1.0
    - np.cos(np.pi * (np.arange(idx1, idx2 + 1) - float(idx1)) / (idx2 - idx1))
  )
  cos_win[idx2 + 1 : idx3] = 1.0
  cos_win[idx3 : idx4 + 1] = 0.5 * (
    1.0
    + np.cos(np.pi * (float(idx3) - np.arange(idx3, idx4 + 1)) / (idx4 - idx3))
  )

  if idx1 == idx2:
    cos_win[idx1] = 0.0
  if idx3 == idx4:
    cos_win[idx3] = 0.0
  return cos_win


def low_pass_filter_fns(
  len_signal, p, corner_frequency=0.05, sampling_frequency=4
):
  """Returns a tapered, demeaned low-pass filter and a no-op inverse.

  Args:
    len_signal: Expected waveform length on axis 1.
    p: Cosine taper fraction passed to ``taper``.
    corner_frequency: Low-pass corner frequency for ``lowpass_filter``.
    sampling_frequency: Sampling rate passed to ``lowpass_filter``.

  Returns:
    A ``(forward, identity)`` pair where ``forward`` filters waveforms and
    ``identity`` returns inputs unchanged.
  """
  tap = taper(len_signal, p).astype(np.float32)
  filt = lowpass_filter(corner_frequency, sampling_frequency).astype(np.float32)

  @tf.numpy_function(Tout=tf.float32)
  def forward(waveform):
    chex.assert_axis_dimension(waveform, 1, len_signal)
    waveform = waveform - waveform.mean(axis=1, keepdims=True)
    waveform = waveform * tap.reshape(1, -1, 1)
    waveform = sosfilt(filt, waveform, axis=1).astype(np.float32)
    return waveform

  return forward, lambda x: x
