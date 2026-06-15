# Hand-to-TeX 🖋️ -> $\TeX$

[![Tests](https://github.com/Projekt-Deep-Learning-2026/hand-to-tex/actions/workflows/python-testing.yml/badge.svg)](https://github.com/Projekt-Deep-Learning-2026/hand-to-tex/actions/workflows/python-testing.yml)
[![Linting](https://github.com/Projekt-Deep-Learning-2026/hand-to-tex/actions/workflows/ruff-linting.yml/badge.svg)](https://github.com/Projekt-Deep-Learning-2026/hand-to-tex/actions/workflows/ruff-linting.yml)
[![Deploy](https://github.com/Projekt-Deep-Learning-2026/hand-to-tex/actions/workflows/deploy.yml/badge.svg)](https://github.com/Projekt-Deep-Learning-2026/hand-to-tex/actions/workflows/deploy.yml)

## Table of Contents

- [Introduction](#introduction)
- [Quick Start Guide](#quick-start-guide)
- [Dataset Details](#dataset-details)
- [Training and Evaluation](#training-and-evaluation)
- [Weights & Biases Logging](#weights--biases-logging)
- [Inference and Demo](#inference-and-demo)
- [Development](#development)

---

## Introduction

Welcome to **Hand-to-TeX**! This is a deep learning project focused on converting online handwritten mathematical expressions (such as InkML stroke data) directly into $\LaTeX$ code. 

**Try it out!** Check out our live web demo here:  
👉 [Hand-to-TeX Web Demo](https://projekt-deep-learning-2026.github.io/hand-to-tex/)

Our project covers the complete pipeline:
- Downloading and preprocessing datasets into efficient `.pt` tensors.
- Training and evaluating deep learning models using PyTorch Lightning.
- Running batch and interactive inferences.

![Web Demo Interface](blank-demo.png)  
*A preview of our interactive web interface showing handwritten math being translated to LaTeX.*

---

## Quick Start Guide

Ready to dive in? Follow these steps to get the project up and running locally.

### 1. Prerequisites

- Python 3.12 or newer.
- The `uv` package manager. You can install it via `pip install uv`.

### 2. Clone and Install Dependencies

First, clone the repository and navigate into the folder:

```bash
git clone https://github.com/Projekt-Deep-Learning-2026/hand-to-tex.git
cd hand-to-tex
```

Next, use `uv` to install the dependencies. Pick the command that matches your hardware setup:

- **Apple Silicon / macOS** (For training & experiments):
  ```bash
  uv sync --dev
  ```
- **CPU-only** (Linux / Windows / macOS - for lightweight inference, no GPU needed):
  ```bash
  uv sync --extra cpu --dev
  ```
- **NVIDIA GPU / CUDA** (Linux / Windows - for training & experiments):
  ```bash
  uv sync --extra gpu --dev
  ```

Install pre-commit hooks to keep code clean:
```bash
uv run pre-commit install
```

### 3. Activate the Virtual Environment

Activating the environment allows you to run commands without prefixing them with `uv run`.

- **macOS / Linux:** `source .venv/bin/activate`
- **Git Bash (Windows):** `source .venv/Scripts/activate`
- **PowerShell (Windows):** `.venv\Scripts\Activate.ps1`


---

## Core CLI Scripts

This project provides several CLI entrypoints installed automatically via `uv`:

- **`htt-init`**: One-command initialization. It downloads the raw data and preprocesses it.
- **`htt-get-data`**: Downloads the raw MathWriting archives.
- **`htt-preprocess`**: Converts raw InkML files into efficient `.pt` tensors for training.
- **`htt-run`**: The main PyTorch Lightning CLI entrypoint used to route `fit` (training) and `test` (evaluation) commands.
- **`htt-demo`**: Runs inference either on provided files/directories or via an interactive drawing canvas.

*Tip: You can append `--help` to any of these commands for a full list of options.*

---

## Dataset Details

This project relies on robust handwritten mathematical data. We primarily use the **MathWriting** dataset.

If you'd like to learn more about the dataset methodology, check out the original research paper:  
📄 [MathWriting: A Database for Online Handwritten Mathematical Expression Recognition](https://arxiv.org/html/2404.10690v1)

### Prepare Data Automatically

To download and preprocess the dataset in one go, simply run:

```bash
htt-init --mode standard --threads 8
```

You can change the `--mode` to `mock` for a fast local check, or `extended` if you want extra merged data.

---

## Training and Evaluation

Training models is straightforward thanks to PyTorch Lightning and the Lightning CLI. We manage configurations using YAML files located in the `configs/` directory.

### Training the Model

To train the model using our default configuration (which expects the data to be in `data/full`), run:

```bash
htt-run fit --config configs/default.yaml
```

If you processed your data into a different folder, you can override the path easily:

```bash
htt-run fit --config configs/default.yaml --data.root data/extended
```

*For quick sanity checks, use `configs/short.yaml` combined with smaller mock data.*


### Evaluating the Model

Once trained, you can evaluate your model's performance on the test set using a checkpoint:

```bash
htt-run test --config configs/default.yaml --ckpt_path checkpoints/last.ckpt
```

---

## Weights & Biases Logging

We use Weights & Biases (W&B) for experiment tracking, monitoring training metrics, and comparing different models.

You can view our public project dashboard and training logs here:  
👉 [Hand-to-TeX W&B Workspace](https://wandb.ai/dl-26-uniwroc-team1/hand-to-tex)

![Validation Progress through training](https://github.com/user-attachments/assets/e865f32d-263a-4cc4-aff8-69ffafc8d73b)  
*Screenshot from W&B. Training with `symbols` split merged into standard dataset and default.yaml config*

---

## Inference and Demo

You can interact with your models in multiple ways using the `htt-demo` CLI tool.

### Batch Inference

Process an entire directory of `.inkml` files at once:

```bash
htt-demo --ckpt data/models/last.ckpt --input data/mathwriting-2024/test
```

To save visualizations of the predictions instead of just viewing them, add the `--save-img` flag:

```bash
htt-demo --ckpt data/models/last.ckpt --input tests/fixtures --save-img
```

### Interactive Canvas Mode

Want to draw your own math expressions? Launch the interactive canvas:

```bash
htt-demo --ckpt data/models/last.ckpt --interactive
```

![Interactive Canvas](blank-canvas.png)  
*The interactive drawing canvas where you can write equations and see live predictions.*

---

## Development

If you are contributing to the project, here are some helpful commands:

**Run tests:**
```bash
pytest
```

**Run linting and formatting:**
```bash
ruff check .
ruff format .
```

Our project structure is organized as follows:
- `configs/` - Training configuration profiles.
- `scripts/` - CLI scripts for data preparation and demos.
- `src/hand_to_tex/` - Core deep learning logic, datasets, and utilities.
- `tests/` - Unit and integration tests.
- `web/` - Web frontend for the browser demo.

---

*This project is licensed under the MIT License. See the `LICENSE` file for details.*
