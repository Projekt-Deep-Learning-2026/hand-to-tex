# Hand-to-TeX

Hand-to-TeX is a deep learning project for converting online handwritten mathematical expressions (InkML stroke data) into LaTeX.

It includes the full workflow:

- dataset download
- preprocessing into efficient `.pt` tensors
- model training and evaluation (PyTorch Lightning)
- batch and interactive inference

## Quick Start

### 1. Prerequisites

- Python 3.12+
- `uv` installed (`pip install uv`)

### 2. Install dependencies

```bash
git clone https://github.com/Projekt-Deep-Learning-2026/hand-to-tex.git
cd hand-to-tex
uv sync --dev
```

### 3. Activate your virtual environment (recommended)

If you activate `.venv`, you can run project commands directly without prefixing them with `uv run`.

**macOS / Linux:**
```bash
source .venv/bin/activate
```
**Windows**
```bash
# Git Bash 
source .venv/Scripts/activate

# PowerShell 
.venv\Scripts\Activate.ps1
```

### 4. Run a prediction immediately (from checkpoint)

You can see how your model performs on .inkml files by running `htt-demo`

```bash
# Provide a file or a directory with .inkmls to visualize sample vs model output
htt-demo --ckpt data/models/last.ckpt --input tests/fixtures/sample.inkml

# Play with interactive window to see how the model recognizes your handwriting
htt-demo --ckpt data/models/last.ckpt --interactive
```




You should see predicted TeX in logs and a rendered plot window.

### 5. Prepare data with one command

```bash
htt-init --mode standard --threads 8
```

### 6. Train

Training setup is managed by Lightning CLI through `htt-run` and YAML configs in `configs/`.

```bash
htt-run fit --config configs/default.yaml
```

### 7. Evaluate

```bash
htt-run test --config configs/default.yaml --ckpt_path checkpoints/last.ckpt
```

## Core CLI Commands

Installed entrypoints:

- `htt-init`: one-command data initialization (download + preprocess)
- `htt-get-data`: download raw MathWriting archives
- `htt-preprocess`: convert InkML to `.pt` tensors
- `htt-run`: train/test with Lightning CLI
- `htt-demo`: run inference on files or interactive canvas

Use `<command> --help` for full options.

## Data Initialization Modes (`htt-init`)

`htt-init` supports three modes via `--mode`:

| Mode | What it does | Output |
|---|---|---|
| `mock` | Downloads excerpt dataset and preprocesses with synthetic/symbol merges | `data/sample` |
| `standard` | Downloads full dataset and preprocesses base splits | `data/full` |
| `extended` | Downloads full dataset and preprocesses with synthetic/symbol merges | `data/extended` |

Examples:

```bash
# Fast local check / small data
htt-init --mode mock --threads 4

# Standard full-data setup
htt-init --mode standard --threads 8

# Extended setup with extra merged data
htt-init --mode extended --threads 8
```

## Manual Data Pipeline

### Download raw data

```bash
# Excerpt dataset
htt-get-data

# Full dataset
htt-get-data --full
```

### Preprocess raw InkML into `.pt`

```bash
htt-preprocess --root data/mathwriting-2024 --out-dir data/full --threads 8
```

Common useful options:

- `--splits train valid test`
- `--merge synthetic symbols`
- `--capacity <N>`
- `--max-tokens <N>`
- `--max-tracepoints <N>`
- `--start-idx <N>`

## Training and Evaluation

This project uses Lightning CLI for training setup, configuration, and command routing (`fit`, `test`).


### Train with the default profile

```bash
htt-run fit --config configs/default.yaml
```

### Train on a different processed dataset root

```bash
htt-run fit --config configs/default.yaml --data.root data/extended
```

### Quick sanity training on smaller data

```bash
htt-run fit --config configs/short.yaml --data.root data/sample --trainer.max_epochs 2
```

### Test a checkpoint

```bash
htt-run test --config configs/default.yaml --ckpt_path checkpoints/last.ckpt
```

## Inference / Demo

### Batch inference from one file

```bash
htt-demo --ckpt data/models/last.ckpt --input tests/fixtures/sample.inkml
```

### Batch inference from directory

```bash
htt-demo --ckpt data/models/last.ckpt --input data/mathwriting-2024/test
```

### Save visualization images

```bash
htt-demo --ckpt data/models/last.ckpt --input tests/fixtures --save-img
```

### Interactive drawing mode

```bash
htt-demo --ckpt data/models/last.ckpt --interactive
```

## Configuration Profiles

- `configs/default.yaml`: main training profile, expects processed data in `data/full`.
- `configs/short.yaml`: lighter profile for short experiments.

Both are standard Lightning CLI configs and can be overridden from command line.

## Development

Run tests:

```bash
pytest
```

Run lint/format checks:

```bash
ruff check .
ruff format .
```

Install pre-commit hooks:

```bash
pre-commit install
```

## Project Structure

```text
configs/                # training profiles
scripts/                # CLI scripts: init/download/preprocess/demo
src/hand_to_tex/        # core package: datasets, model, utils, runtime CLI
tests/                  # unit tests
data/                   # raw and processed datasets, checkpoints
```

## License

MIT. See `LICENSE`.
