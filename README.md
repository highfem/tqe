# This quake exists

[![ci](https://github.com/highfem/tqe/actions/workflows/ci.yaml/badge.svg)](https://github.com/highfem/tqe/actions/workflows/ci.yaml)
[![arXiv](https://img.shields.io/badge/arXiv-2610.04334-b31b1b.svg)](https://arxiv.org/abs/2610.04334)

## About

`tqe` implements *HighFEM-2*, a Generative Waveform Model (GWM) for conditional
synthesis of seismic waveforms built on JAX and Flax. For details see the
[manuscript](https://arxiv.org/abs/2610.04334).

The repository has two layers:

- **`src/tqe/`** — the installable library (diffusion, flow matching, VAE,
  WGAN objectives; signal processing pipeline).
- **`experiments/`** — training and evaluation code used to produce the
  manuscript results.

## Quick start

Install `tqe` from the latest release:

```bash
pip install git+https://github.com/highfem/tqe@<RELEASE>
```

Replace `<RELEASE>` with the desired tag from the
[releases page](https://github.com/highfem/tqe/tags). For GPU support, include the CUDA extra:

```bash
pip install "git+https://github.com/highfem/tqe@<RELEASE>#egg=tqe[cuda]"
```

### Generating waveforms

Pretrained weights are downloaded automatically from [Zenodo](https://doi.org/10.5281/zenodo.23107915) on first use (cached to `~/.cache/tqe/weights/`; override with `$TQE_CACHE_DIR`).
The command reads an HDF5 file of context waveforms and writes synthetic target waveforms to a new HDF5 file:

```bash
generate-waveforms \
  --input  path/to/input.h5  \
  --output path/to/output.h5
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-per-entry` | `1` | Waveforms to generate per input entry |
| `--batch-size-per-device` | `64` | Per-device batch size |
| `--seed` | `0` | RNG seed for reproducibility |
| `--weights-dir` | *(auto)* | Use a local weights directory (skips download) |

The input HDF5 must contain a `context` dataset of shape `(N, L, 3)` plus
scalar metadata fields (`magnitude`, `vs30`, `station_longitude`, etc.).
The output contains `target_hat`, `context`, and `meta` datasets.

## Experiments

To replicate the manuscript experiments (data preprocessing, training,
evaluation), see [experiments/amatrice](./experiments/amatrice/).

## Development

Install [uv](https://docs.astral.sh/uv/), then sync all dependencies:

```bash
uv sync --all-extras
```

**Lint and format**

```bash
uv run ruff check --fix src/tqe
uv run ruff format src/tqe
```

**Type check**

```bash
uv run mypy src/tqe
```

**Tests**

```bash
uv run pytest tests/
```

**Pre-commit** (runs ruff, gitlint, and other checks on every commit)

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run pre-commit run --all-files   # run manually across all files
```

## Contributing

We welcome contributions in the form of pull requests. To get started:

1. **Clone the repository**

   ```bash
   git clone https://github.com/highfem/tqe
   ```

2. **Install dependencies and set up pre-commit hooks**

   ```bash
   uv sync --all-extras
   uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
   ```

3. **Create a branch**

   ```bash
   git checkout -b feature/my-new-feature   # new feature
   git checkout -b issue/fixes-bug          # bug fix
   ```

4. **Implement your change and verify**

   ```bash
   uv run ruff check --fix src/tqe
   uv run mypy src/tqe
   uv run pytest tests/
   ```

5. **Push your branch and open a pull request.**

## Citation

If you find our work relevant to your research, please consider citing:

```bibtex
@misc{palgunadi2026broadband,
    title={Broadband Ground-Motion Synthesis by Conditioning Denoising Diffusion Models on Low-Frequency Waveforms},
    author={Kadek Hendrawan Palgunadi and Simon Dirmeier and Maria Koroni and Laura Ermert and Men-Andrin Meier},
    year={2026},
    eprint={2610.04334},
    archivePrefix={arXiv},
    primaryClass={physics.geo-ph},
    url={https://arxiv.org/abs/2610.04334}
}
```
