# hand-to-tex

Model for converting online handwritten mathematical expressions into LaTeX.

## Overview

The project is built around:

1. Data preprocessing from InkML into tensors.
2. Transformer-based sequence modeling.
3. Training and evaluation with Lightning CLI and Weights & Biases logging.

## Local Setup

This repository uses uv for dependency management.

1. Clone repository.

```bash
git clone https://github.com/Projekt-Deep-Learning-2026/hand-to-tex.git
cd hand-to-tex
```

2. Install dependencies.

```bash
uv sync --dev
```

3. Install pre-commit hooks.

```bash
uv run pre-commit install
```

## Dataset

The project uses Google MathWriting (InkML-based online handwriting).

Important raw splits include:

1. train
2. valid
3. test
4. symbols
5. synthetic

Use normalized labels as training targets.

## Download Data

Small sample download:

```bash
uv run htt-get-data
```

Full dataset download:

```bash
uv run htt-get-data --full
```

## Preprocess Data

Preprocessing converts InkML files to ready-to-train tensor samples stored in .pt files.

Run preprocessing:

```bash
uv run htt-preprocess --threads 8
```

Common options:

1. `--root`: source dataset root (default: `data/mathwriting-2024`)
2. `--vocab`: vocabulary path (default: `data/assets/vocab.json`)
3. `--threads`: number of worker processes
4. `--splits`: selected splits to preprocess
5. `--merge`: selected additional data to merge into created splits
6. `--out-dir`: output directory for generated .pt files
7. `--capacity`: limit number of kept samples per split
8. `--max-tokens`: filter out long token sequences
9. `--max-tracepoints`: filter out long traces

Examples:

```bash
uv run htt-preprocess --threads 12 --splits train valid test 
uv run htt-preprocess --threads 4 --splits test
```

## Training Config Profiles

Configuration profiles are stored in `configs/`.

1. `configs/default.yaml`: full training profile (`data/full`)
2. `configs/short.yaml`: short profile for faster experiments (`data/shorter`)

## Run Training

Train with full profile:

```bash
uv run python main.py fit --config configs/default.yaml
```

Train with short profile:

```bash
uv run python main.py fit --config configs/short.yaml
```

Override values from CLI (example):

```bash
uv run python main.py fit --config configs/short.yaml --trainer.max_epochs 5 --data.batch_size 64
```

Run test with a checkpoint:

```bash
uv run python main.py test --config configs/default.yaml --ckpt_path path/to/checkpoint.ckpt
```

## Development Commands

Run tests:

```bash
uv run pytest
```

Lint and format checks:

```bash
uv run ruff check .
uv run ruff format .
```

## Notes

1. The training module logs text metrics and exact-match ratio on validation and test.
2. WandB logger is configured through Lightning config files.
3. If you change dataset paths or vocab, update the selected config profile.
