# hand-to-tex

Model for converting online handwritten mathematical expressions into LaTeX.

## Quick Start

This repository uses uv for dependency management.

1. Clone the repository.

```bash
git clone https://github.com/Projekt-Deep-Learning-2026/hand-to-tex.git
cd hand-to-tex
```

2. Install dependencies.

```bash
uv sync --dev
```

3. (Optional) install pre-commit hooks.

```bash
uv run pre-commit install
```

4. Initialize full data pipeline (download + preprocess).

```bash
uv run htt-init --threads 8
```

By default this runs:

1. `htt-get-data --full`
2. `htt-preprocess --threads <N> --out-dir data/full --merge synthetic symbols`

5. For a quick mock/sample setup use:

```bash
uv run htt-init --mock --threads 4
```

Mock mode runs excerpt download and preprocesses into `data/sample`.

6. Train a baseline model.

```bash
uv run htt-run fit --config configs/default.yaml
```

## Scripts

### htt-init

One-command initialization for dataset preparation.

Options:

1. `--threads`: number of worker processes passed to preprocessing (default: `1`)
2. `--mock`: use excerpt dataset instead of full dataset

Examples:

```bash
uv run htt-init
uv run htt-init --threads 8
uv run htt-init --mock --threads 4
```

### htt-get-data

Download dataset archives.

Options:

1. `--full`: download full MathWriting dataset

Examples:

```bash
uv run htt-get-data
uv run htt-get-data --full
```

### htt-preprocess

Convert InkML files to tensor samples in `.pt` format.

Common options:

1. `--root`: source dataset root (default: `data/mathwriting-2024`)
2. `--vocab`: vocabulary path (default: `data/assets/vocab.json`)
3. `--threads`: number of worker processes
4. `--splits`: selected splits to preprocess (`train`, `valid`, `test`)
5. `--merge`: additional folders from `--root` to merge into selected splits
6. `--out-dir`: output directory for generated `.pt` files
7. `--start-idx`: skip first N files
8. `--capacity`: limit number of kept samples per split
9. `--max-tokens`: filter out long token sequences
10. `--max-tracepoints`: filter out long trace sequences

Examples:

```bash
uv run htt-preprocess --threads 12 --splits train valid test
uv run htt-preprocess --threads 4 --splits test
uv run htt-preprocess --threads 8 --out-dir data/full --merge synthetic symbols
```

### htt-run

Entry point for the project runtime CLI.

## Dataset

The project uses Google MathWriting (InkML-based online handwriting).

Important raw splits include:

1. train
2. valid
3. test
4. symbols
5. synthetic

Use normalized labels as training targets.

## Training

Configuration profiles are stored in `configs/`.

1. `configs/default.yaml`: full training profile (`data/full`)
2. `configs/short.yaml`: short profile for faster experiments (`data/shorter`)

Train with full profile:

```bash
uv run htt-run fit --config configs/default.yaml
```

Train with short profile:

```bash
uv run htt-run fit --config configs/short.yaml
```

Override values from CLI (example):

```bash
uv run htt-run fit --config configs/short.yaml --trainer.max_epochs 5 --data.batch_size 64
```

Run test with a checkpoint:

```bash
uv run htt-run test --config configs/default.yaml --ckpt_path path/to/checkpoint.ckpt
```

## Weights & Biases (WandB)

Project dashboard:

- https://wandb.ai/dl-26-uniwroc-team1/hand-to-tex


## Development

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
3. If you change dataset paths or vocabulary, update the selected config profile.
