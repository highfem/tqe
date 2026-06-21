"""Generative waveform modeling (JAX / Flax)."""

from tqe._types import ObjectiveFns, TrainFns
from tqe.autoencoder import AutoencoderConfig, autoencoder
from tqe.classifier import ClassifierConfig, classifier
from tqe.denoising_diffusion import EDMParameterization, denoising_diffusion
from tqe.diffusion_consistency_distillation import (
  EDMConsistencyDistillationConfig,
  diffusion_consistency_distillation,
)
from tqe.flow_matching import FlowMatchingConfig, flow_matching
from tqe.wgan import WGANConfig, WGANState, wgan

__all__ = [
  "autoencoder",
  "AutoencoderConfig",
  "ClassifierConfig",
  "classifier",
  "denoising_diffusion",
  "diffusion_consistency_distillation",
  "EDMConsistencyDistillationConfig",
  "EDMParameterization",
  "flow_matching",
  "FlowMatchingConfig",
  "ObjectiveFns",
  "TrainFns",
  "wgan",
  "WGANConfig",
  "WGANState",
]

__version__ = "0.1.0"
