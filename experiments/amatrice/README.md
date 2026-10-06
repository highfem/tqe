# Amatrice experiment

This directory contains the training and evaluation code for *HighFEM-2*,
a Generative Waveform Model (GWM) trained on the Amatrice dataset.
See the [manuscript](https://arxiv.org/abs/2610.04334) for details.

## Setup

A fully-pinned conda environment is provided for reproducibility:

```bash
conda env create -f envs/environment.yaml -p <ENVDIR>
conda activate <ENVDIR>
```

`<ENVDIR>` is the path where the environment will be installed (e.g.
`/path/to/envs/tqe`).

## Data

The TFDS dataset is built from the raw Amatrice paired-waveform files.
From this directory:

```bash
cd dataset/amatrice_paired_waveforms
python amatrice_paired_waveforms_dataset_builder.py
```

Set the raw data path and output directory inside the builder script before
running (or pass them as arguments if you have added CLI support).

## Training

All experiments are driven through `main.py`.

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *(required)* | Config file path |
| `--workdir` | *(required)* | Output directory for checkpoints and logs |
| `--experiment` | *(required)* | `autoencoder`, `latent_diffusion`, `latent_wgan`, `latent_consistency_distillation`, `classifier` |
| `--mode` | `train` | `train` or `evaluate` |
| `--usewand` | `False` | Log run to Weights & Biases |
| `--batch-size-per-device` | `64` | Per-device batch size |
| `--autoencoder_checkpoints` | `target_encoder,context_encoder` | Comma-separated checkpoint names for autoencoder submodels |
| `--autoencoder_checkpoints_steps` | `best,best` | Checkpoint step for each autoencoder (`best` or a step number) |
| `--model_checkpoint` | `latent_diffusion` | Checkpoint name of the generative model to evaluate |
| `--classifier_checkpoint` | `classifier` | Checkpoint name of the classifier used to compute embeddings |
| `--outfile` | — | Output HDF5 path (evaluate mode only) |
| `--ddm_checkpoint` | — | Pre-trained DDM checkpoint name (consistency distillation only) |
| `--ddm_checkpoint_step` | — | Step of the DDM checkpoint (consistency distillation only) |

### Step 1 — Autoencoders

Train a target encoder and a context encoder separately using
`config-autoencoder.py`. The `model_type` config key controls which one
is trained.

```bash
# target encoder
python main.py \
  --config=configs/config-autoencoder.py \
  --config.model_type=target_encoder \
  --workdir=<WORKDIR> \
  --experiment=autoencoder \
  --mode=train

# context encoder
python main.py \
  --config=configs/config-autoencoder.py \
  --config.model_type=context_encoder \
  --workdir=<WORKDIR> \
  --experiment=autoencoder \
  --mode=train
```

### Step 2 — Latent diffusion model

Train a latent DDM on top of the frozen autoencoder representations.
Pass the autoencoder checkpoint names via `--autoencoder_checkpoints`.

**Flow matching with DiT:**

```bash
python main.py \
  --config=configs/config-ddm-fm-dit.py \
  --workdir=<WORKDIR> \
  --experiment=latent_diffusion \
  --mode=train \
  --autoencoder_checkpoints=target_encoder,context_encoder \
  --autoencoder_checkpoints_steps=best,best
```

**EDM with consistency distillation:**

```bash
python main.py \
  --config=configs/config-ddm-edm-dit.py \
  --workdir=<WORKDIR> \
  --experiment=latent_diffusion \
  --mode=train \
  --autoencoder_checkpoints=target_encoder,context_encoder \
  --autoencoder_checkpoints_steps=best,best
```

```bash
python main.py \
  --config=configs/config-cd-edm-dit.py \
  --workdir=<WORKDIR> \
  --experiment=latent_consistency_distillation \
  --mode=train \
  --autoencoder_checkpoints=target_encoder,context_encoder \
  --autoencoder_checkpoints_steps=best,best \
  --ddm_checkpoint=latent_diffusion \
  --ddm_checkpoint_step=<STEP>
```

### Step 3 — Evaluation

Evaluate mode generates synthetic waveforms from the test split, runs them
through a pretrained classifier to extract embeddings, and writes everything
to an HDF5 file for downstream metric computation. The output contains
`target`, `target_hat`, `context`, `meta`, plus classifier embeddings and
predictions for both real and generated waveforms.

**Latent diffusion (flow matching or EDM):**

```bash
python main.py \
  --config=configs/config-ddm-fm-dit.py \
  --workdir=<WORKDIR> \
  --experiment=latent_diffusion \
  --mode=evaluate \
  --outfile=<WORKDIR>/generated.h5 \
  --model_checkpoint=latent_diffusion \
  --classifier_checkpoint=classifier \
  --autoencoder_checkpoints=target_encoder,context_encoder \
  --autoencoder_checkpoints_steps=best,best
```

`--model_checkpoint` and `--classifier_checkpoint` match the checkpoint
subdirectory names under `<WORKDIR>/checkpoints/` and can be omitted if
the defaults (`latent_diffusion` and `classifier`) apply.

**Consistency distillation:**

```bash
python main.py \
  --config=configs/config-cd-edm-dit.py \
  --workdir=<WORKDIR> \
  --experiment=latent_diffusion \
  --mode=evaluate \
  --outfile=<WORKDIR>/generated-cd.h5 \
  --model_checkpoint=latent_consistency_distillation \
  --classifier_checkpoint=classifier \
  --autoencoder_checkpoints=target_encoder,context_encoder \
  --autoencoder_checkpoints_steps=best,best
```

## Available configs

| Config file | Experiment | Architecture |
|-------------|-----------|--------------|
| `config-autoencoder.py` | `autoencoder` | VAE (target or context encoder) |
| `config-ddm-fm-dit.py` | `latent_diffusion` | Flow matching + DiT |
| `config-ddm-edm-dit.py` | `latent_diffusion` | EDM + DiT |
| `config-cd-edm-dit.py` | `latent_consistency_distillation` | EDM consistency distillation + DiT |
| `config-wgan-pix2pix.py` | `latent_wgan` | WGAN-GP + Pix2Pix |
