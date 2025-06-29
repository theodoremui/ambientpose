# Using `uv` for Python Package Management in AlphaDetect

This document provides a comprehensive guide to setting up and using `uv`, the extremely fast Python package installer and resolver, for the AlphaDetect project. `uv` is used to manage virtual environments and dependencies, replacing traditional tools like `pip` and `venv`.

---

## Table of Contents

1.  [Why `uv`?](#1-why-uv)
2.  [Installation](#2-installation)
3.  [Core Concepts](#3-core-concepts)
4.  [Setting Up the AlphaDetect Project](#4-setting-up-the-alphadetect-project)
5.  [Daily Development Workflow](#5-daily-development-workflow)
    - [Viewing Installed Packages](#viewing-installed-packages)
    - [Adding a Dependency](#adding-a-dependency)
    - [Removing a Dependency](#removing-a-dependency)
    - [Updating Dependencies](#updating-dependencies)
6.  [Virtual Environments with `uv`](#6-virtual-environments-with-uv)
7.  [Best Practices](#7-best-practices)
8.  [Troubleshooting](#8-troubleshooting)

---

## 1. Why `uv`?

The AlphaDetect project uses `uv` for its Python environment and package management for several key reasons:

-   **Speed**: `uv` is written in Rust and is significantly faster than `pip` and `venv`, especially when installing from a lock file or resolving complex dependency trees.
-   **All-in-One Tool**: It combines the functionality of `pip`, `venv`, `pip-tools`, and more into a single, cohesive command-line interface.
-   **Modern Foundation**: It leverages modern Python packaging standards (PEP 517/518) and `pyproject.toml`.
-   **Reproducibility**: `uv` makes creating and maintaining lock files for reproducible builds straightforward.

> **AlphaDetect convention**  
> * `pyproject.toml` is the **single source-of-truth** for all direct
>   dependencies.  
> * `requirements.txt` is **auto-generated** (via
>   `uv pip compile`) and functions only as a *lock-file* that pins
>   exact versions for CI / production – do **not** edit it manually.

## 2. Installation

Install `uv` on your system. It's a standalone binary with no dependencies.

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

After installation, verify it was successful:

```bash
uv --version
```

## 3. Core Concepts

`uv` provides a set of commands that will feel familiar if you've used `pip` and `venv`.

-   `uv venv`: Creates and manages virtual environments.
-   `uv pip install`: Installs packages into a virtual environment.
-   `uv pip uninstall`: Uninstalls packages.
-   `uv pip sync`: Synchronizes the environment to match the exact contents of a lock file (`uv.lock` or `requirements.txt`).
-   `uv pip compile`: Resolves dependencies from `pyproject.toml` and **writes a
    fully-pinned `requirements.txt`** lock file.
-   `uv run`: Runs a command within the managed virtual environment without needing to activate it first.

## 4. Setting Up the AlphaDetect Project

Follow these steps to set up your local development environment for AlphaDetect using `uv`.

**Step 1: Clone the Repository**

```bash
git clone https://github.com/your-org/alphadetect.git
cd alphadetect
```

**Step 2: Create the Virtual Environment**

Use `uv venv` to create a virtual environment named `.venv` in the project root. `uv` will automatically use the Python version specified in the `.python-version` file if available.

```bash
uv venv
```

This creates a `.venv` directory. You can activate it like any other virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**Step 3: Install Dependencies**

Install all project dependencies, including the optional `dev` and `test` groups, directly from `pyproject.toml`. The `-e` flag installs the `alphadetect` project in "editable" mode.

```bash
uv pip install -e ".[dev,test]"
```

**For GPU-accelerated computation**, install the `gpu` extra, which includes CUDA-enabled PyTorch wheels:

```bash
uv pip install -e ".[dev,test,gpu]"
```

**Step 4 (once base env is ready): Install AlphaPose**

AlphaPose is *not* listed in `pyproject.toml` because its build step
requires **NumPy to be present first**.  After completing the previous
step, install AlphaPose manually:

```bash
# inside the activated .venv  – or use `uv run`
uv pip install "alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose"
```

> If AlphaPose is missing, `cli/detect.py` will print a helpful error
> message instructing you to run the command above.

Your environment is now fully set up and ready for development.

## 5. Daily Development Workflow

Here’s how to manage dependencies during your day-to-day work on AlphaDetect.

### Viewing Installed Packages

To see what's installed in your `.venv`, use `uv pip list`:

```bash
uv pip list
```

### Adding a Dependency

1.  **Install the new package**:
    Let's say you need to add `scikit-learn`.

    ```bash
    uv pip install scikit-learn
    ```

2.  **Update `pyproject.toml`**:
    Manually add the new package to the `dependencies` list in your `pyproject.toml` file. This keeps your project's requirements explicit.

    ```toml
    # pyproject.toml
    ...
    dependencies = [
      "fastapi>=0.110",
      ...
      "scikit-learn>=1.4" # Add the new package here
    ]
    ```

3.  **Update the Lock File**:
    Re-compile the dependencies to update the **lock file
    (`requirements.txt`)** so that CI and other developers get the exact same versions.

    ```bash
    uv pip compile pyproject.toml -o requirements.txt --all-extras
    ```

### Removing a Dependency

1.  **Uninstall the package**:

    ```bash
    uv pip uninstall scikit-learn
    ```

2.  **Update `pyproject.toml`**:
    Remove the corresponding line from the `dependencies` list.

3.  **Update the Lock File**:
    Re-compile to remove the package from `requirements.txt`.

    ```bash
    uv pip compile pyproject.toml -o requirements.txt --all-extras
    ```

### Updating Dependencies

To update a specific package to its latest compatible version:

```bash
uv pip install --upgrade scikit-learn
```

To update all packages according to the versions specified in `pyproject.toml`, you can re-run the installation:

```bash
uv pip install -e ".[dev,test]" --upgrade
```

After updating, remember to re-compile the lock file.

## 6. Virtual Environments with `uv`

While activating the virtual environment with `source .venv/bin/activate` is common, `uv` provides a convenient way to run commands without activation using `uv run`:

```bash
# Run pytest without activating the venv
uv run pytest

# Run the CLI without activating the venv
uv run alphadetect-detect --video path/to/video.mp4
```

This is particularly useful for one-off commands or in scripts where activation can be cumbersome.

## 7. Best Practices

-   **Commit `uv.lock`**: Always commit the `uv.lock` file to your Git repository. This file guarantees that every developer and every CI run uses the exact same versions of all dependencies, preventing "works on my machine" issues.
-   **Commit `requirements.txt`**: Always commit the generated
    `requirements.txt` lock file. It guarantees that every developer and
    every CI run uses the exact same versions, preventing "works on my
    machine" issues.
-   **Use `uv pip sync` in CI**: For CI/CD pipelines, use
    `uv pip sync requirements.txt` instead of `uv pip install`. This
    skips dependency resolution and installs directly from the lock
    file, making builds deterministic **and** fast.
-   **Define Dependencies in `pyproject.toml`**: Your `pyproject.toml` file should always be the single source of truth for your project's direct dependencies.
-   **Use Optional Dependency Groups**: Organize dependencies into groups like `[project.optional-dependencies]` for `dev`, `test`, and `gpu` to keep the core installation lean.

## 8. Troubleshooting

-   **`uv: command not found`**: Your shell hasn't picked up `uv`'s location. Make sure `~/.cargo/bin` (for Linux/macOS) or `%USERPROFILE%\.cargo\bin` (for Windows) is in your `PATH` environment variable, or restart your terminal.
-   **Installation Failures**: If a package fails to build, it might be missing system-level dependencies (e.g., a C compiler, or libraries like `libpq-dev` for PostgreSQL). Check the error logs for clues.
-   **Clearing the Cache**: `uv` maintains a global cache to speed up installations. If you suspect a corrupted package is causing issues, you can clear the cache:
    ```bash
    uv cache clean
    ```

---

By following this guide, you can leverage the full power of `uv` to maintain a fast, clean, and reproducible development environment for the AlphaDetect project.
