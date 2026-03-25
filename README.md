## 🛠 Local Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [Ruff](https://github.com/astral-sh/ruff) for code formatting and linting.

To start working on the code:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Projekt-Deep-Learning-2026/hand-to-tex.git
   cd hand-to-tex

2. **Install development dependencies:**
   ```bash
   uv sync --dev
   ```

3. **Enable pre-commit hooks:**
   This is required to ensure your code is automatically formatted and linted before every commit.
   ```bash
   uv run pre-commit install
   ```

From now on, you can use `git commit` normally. If Ruff finds errors or auto-formats your files, your commit will be interrupted. When this happens, simply stage the automatically modified files (`git add .`) and run your `git commit` command again.

Please update this readme section in case of adding new tools for development that require initial setup!