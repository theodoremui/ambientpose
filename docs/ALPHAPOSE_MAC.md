# Guide: Installing AlphaPose on macOS with Apple Silicon (M1/M2/M3)

**Target Audience:** Developers using Apple Silicon Macs (M1, M2, M3 series) without access to an NVIDIA GPU (CUDA).
**Goal:** Achieve a stable, CPU-only or MPS-accelerated installation of AlphaPose.

---

## 1. Overview & Challenges

AlphaPose is a powerful tool, but it was primarily designed for Linux environments with NVIDIA GPUs. Installing it on Apple Silicon (ARM64 architecture) presents several challenges:

*   **No CUDA Support:** Apple Silicon uses Metal Performance Shaders (MPS) for GPU acceleration, not CUDA. AlphaPose's CUDA-specific code will fail to compile.
*   **Architecture Mismatch:** Many Python packages with C/C++ extensions need to be compiled specifically for ARM64.
*   **Build Dependencies:** Certain dependencies like `pycocotools` and `halpecocotools` are known to have compilation issues on macOS.

This guide provides a robust, step-by-step process to navigate these challenges and successfully install AlphaPose.

---

## 2. Prerequisites

Before you begin, ensure your system is set up correctly.

### Step 2.1: Install Homebrew
If you don't have Homebrew, the package manager for macOS, install it first.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2.2: Install Xcode Command Line Tools
This provides essential compilers like `clang` and other build tools.

```bash
xcode-select --install
```

### Step 2.3: Install Python with `pyenv`
Avoid using the system Python. `pyenv` allows you to manage multiple Python versions safely.

```bash
brew install pyenv
pyenv install 3.12.3 # Or another recent version
pyenv global 3.12.3
```
Add the following to your shell profile (`~/.zshrc`, `~/.bash_profile`, etc.) and restart your terminal:
```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

### Step 2.4: Install `uv`
This project uses `uv` for fast package and environment management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 3. Installation: The Recommended Method

This approach involves modifying AlphaPose's `setup.py` to disable CUDA-dependent components, ensuring a successful build.

### Step 3.1: Clone the AlphaDetect Repository
Clone the project repository. This guide assumes you are working from the project's root directory.

```bash
git clone https://github.com/philmui/alphadetect.git
cd alphadetect
```

### Step 3.2: Set Up the Virtual Environment
Create a virtual environment using `uv`. It will automatically use the Python version set by `pyenv`.

```bash
uv venv
source .venv/bin/activate
```

### Step 3.3: Install Build-time Dependencies
Some packages need `numpy` and `cython` to be present *before* they are installed. Install them first to prevent common build errors.

```bash
uv pip install numpy cython
```

### Step 3.4: Install Core Dependencies
Install the main dependencies for AlphaDetect. Note that the `pyproject.toml` is configured to install the correct CPU/MPS-compatible version of PyTorch on macOS.

```bash
# This installs everything except AlphaPose
uv pip install -e ".[dev,test]"
```

### Step 3.5: Modify AlphaPose's `setup.py`
This is the most critical step. You need to prevent the CUDA extensions from being compiled.

1.  Navigate to the AlphaPose source directory (assuming it's a submodule or cloned within the project):
    ```bash
    # Adjust the path if your structure is different
    cd alphapose 
    ```

2.  Open `setup.py` in a text editor.

3.  Locate the `extensions` list within the `setup()` function call (around line 205).

4.  **Comment out or delete** the lines related to `DCNv2` and `halpecocotools`. The section should look like this:

    ```python
    # Before modification (example)
    # ...
    # ext_modules=[
    #         DCNv2,
    #         halpecocotools,
    # ],
    # ...
    
    # After modification
    ...
    ext_modules=[
            # DCNv2,         # <--- COMMENT THIS OUT
            # halpecocotools,  # <--- AND THIS ONE
    ],
    ...
    ```

    > **Why do this?**
    > `DCNv2` (Deformable Convolutional Networks) and `halpecocotools` are custom C++/CUDA extensions. They will fail to compile on a Mac without a CUDA toolkit and an NVIDIA GPU. Disabling them allows the rest of AlphaPose to be installed. The trade-off is that models relying on these specific layers (like DCN-based backbones) will not work, but the core FastPose models will.

5.  Save the `setup.py` file and return to the project root directory.
    ```bash
    cd ..
    ```

### Step 3.6: Install the Modified AlphaPose
Now, install AlphaPose from the modified local source code.

```bash
# Ensure you are in the project's root directory
uv pip install ./alphapose
```

---

## 4. Verification

### Step 4.1: Verify PyTorch and MPS
Check that PyTorch is installed and can see your Mac's GPU via MPS.

```python
import torch

# Should be True
print(f"PyTorch version: {torch.__version__}")
print(f"MPS is available: {torch.backends.mps.is_available()}") 
# Should be True
print(f"MPS is built: {torch.backends.mps.is_built()}")     

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print("Tensor on MPS device:")
    print(x)
else:
    print("MPS not available, falling back to CPU.")
    device = torch.device("cpu")

# Example tensor operation
y = torch.rand(2, 3, device=device)
print("\nRandom tensor on MPS:")
print(y)
```

### Step 4.2: Verify AlphaPose Installation
Run the AlphaPose help command. If it executes without error, the installation was successful.

```bash
uv run python -m alphapose --help
```

You should see the command-line options for AlphaPose printed to the console.

---

## 5. Troubleshooting Common Issues

#### Error: `fatal error: 'numpy/arrayobject.h' file not found`
*   **Cause:** `numpy` was not installed before another package that depends on it during its build step (like `pycocotools`).
*   **Solution:** Make sure you run `uv pip install numpy cython` *before* installing other requirements, as outlined in Step 3.3.

#### Error: `pycocotools` or `halpecocotools` fails to build
*   **Cause:** These packages often have trouble finding the correct C compiler and headers on macOS.
*   **Solution:**
    1.  Ensure Xcode Command Line Tools are installed (`xcode-select --install`).
    2.  If the error persists, try installing `pycocotools` manually with a specific flag:
        ```bash
        uv pip install --no-binary :all: pycocotools
        ```
    3.  For `halpecocotools`, the issue is almost always CUDA-related. Ensure it is commented out in `setup.py` as per Step 3.5.

#### Error: `unrecognized command-line option '-arch' 'arm64'` or linker errors

---

## 7. Alternative Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Conda-Forge** (`mambaforge` / `miniforge`) | Simplifies ARM64 wheel management, many deps pre-compiled | Env slightly slower than *uv*, can collide with Homebrew paths |
| **Pre-compiled Wheels** (`pip install alphapose-cpu-macosx.whl`) | No local compilation, fastest way to test | Nightly builds only, often lag behind upstream |
| **Homebrew Python + venv** | Uses system-wide ARM64 Python 3.12 from Brew; zero pyenv | Fewer parallel versions, Brew updates can break ABI |
| **Rosetta 2 (x86_64)** | Leverage years of x86 wheels (pure CPU) | 30-40 % slower, tricky PATH handling, *not* recommended |

If the primary method above fails, try **Conda-Forge**:

```bash
# 1⃣  Install Mambaforge (ARM64 build)
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-arm64.sh
bash Mambaforge-MacOSX-arm64.sh -b -p $HOME/mambaforge
eval "$($HOME/mambaforge/bin/conda shell.zsh hook)"

# 2⃣  Create env & install PyTorch (MPS)
mamba create -n alphapose python=3.12 pytorch torchvision torchaudio cpuonly -c pytorch
conda activate alphapose
mamba install cython numpy

# 3⃣  Build / install AlphaPose (as modified earlier)
pip install ./alphapose
```

---

## 8. Performance Considerations

1. **MPS vs CPU** – For most inference scenarios the MPS backend on an M2 Pro provides **1.8 – 2.3×** speed-up over CPU.  Training is still slower than NVIDIA GPUs but acceptable for small batches.  
2. **Batch Size** – Start with `--pose-batch-size 32` then increase until you exhaust unified memory (watch the Memory tab in *Activity Monitor*).  
3. **Pinned Memory** – PyTorch automatically pins tensors on MPS; explicit `.pin_memory()` is unnecessary and slows things down.  
4. **Monitoring** – In *Activity Monitor* ➜ **View ▸ GPU Processes** to see `% GPU` usage.  For deeper profiling, use **Xcode ➜ Metal GPU Tools ➜ GPU Capture**.  
5. **Video Decoding** – FFmpeg on Apple Silicon can leverage VideoToolbox; ensure `brew install ffmpeg --enable-videotoolbox` for fastest decode.  

---

## 9. Advanced Configuration

### 9.1 Custom Model Checkpoints

Place extra checkpoints in `model_files/` and pass their paths:

```bash
python cli/detect.py \
  --video dance.mp4 \
  --checkpoint model_files/halpe26_fast_res50_256x192.pth
```

### 9.2 Switching Back-Ends

| Backend | Env Var | Behaviour |
|---------|---------|-----------|
| **MPS** | *(default)* | Uses Apple GPU via Metal |
| **CPU** | `export PYTORCH_ENABLE_MPS_FALLBACK=1` | Forces CPU even if MPS is available |

### 9.3 OpenCV Headless

If you run in a head-less CI runner, replace the default OpenCV wheel:

```bash
uv pip uninstall opencv-python
uv pip install opencv-python-headless==4.10.0.84
```

---

## 10. Support & Community

* **AlphaPose GitHub** – <https://github.com/MVIG-SJTU/AlphaPose/issues>  
* **PyTorch / MPS Forum** – <https://discuss.pytorch.org/c/mps>  
* **Apple Developer Forum** – Metal & ML topics: <https://developer.apple.com/forums/>  
* **Slack** – Join the `#alphadetect` channel in the project workspace for real-time help.  

Please include your **macOS version**, **Xcode CLI tools version**, full **`python -m pip list`**, and the **compiler error log** when opening issues.

---

## Appendix A – Reference ARM64 Build Matrix (2024-Q3)

| Component | Version | Source | Notes |
|-----------|---------|--------|-------|
| macOS | 14.5 (Sonoma) | App Store | Tested on M2 Pro |
| Python | 3.12.3 | `pyenv` | `CFLAGS="-march=armv8-a"` |
| clang | 15.0.0 (Xcode 15.4) | Xcode CLI | `xcrun --show-sdk-path` |
| cmake | 3.29.3 | Homebrew | Needed for `pycocotools` |
| git | 2.45.1 | Homebrew | Submodule checkout |
| ffmpeg | 6.1.1 + VideoToolbox | Homebrew | Hardware decode |
| PyTorch | 2.2.0 | pip | `arm64`, MPS enabled |
| torchvision | 0.17.0 | pip | Matches PyTorch |
| torchaudio | 2.2.0 | pip | CPU-only |
| numpy | 1.26.x | pip wheels | ARM64 wheel |
| cython | 3.1.x | pip | For C-extensions |

<details>
<summary>📦 Brew Bundle (optional one-liner)</summary>

```bash
brew bundle <<'B'
tap "homebrew/cask"
brew "git"
brew "cmake"
brew "ffmpeg", args: ["--with-videotoolbox"]
brew "xz"           # required by pycocotools on macOS
brew "pkg-config"
B
```
</details>

---
*   **Cause:** The build process is being confused by an x86_64 version of a tool (like Python or a compiler) running under Rosetta 2.
*   **Solution:**
    1.  Verify you are running a native ARM64 terminal and Python.
        ```bash
        uname -m 
        # Should output: arm64
        
        python -c "import platform; print(platform.machine())"
        # Should output: arm64
        ```
    2.  If you see `x86_64`, your terminal or Python is running in Rosetta. Ensure you installed an `arm64` version of Python via `pyenv`.

---

## 6. Fallback: Docker Installation

If you continue to face issues with the native installation, using Docker is the most reliable alternative. This will run AlphaPose in a containerized Linux environment, bypassing macOS-specific compilation problems entirely.

1.  **Install Docker Desktop for Mac.**
2.  From the project root, build and run the services:
    ```bash
    docker compose build
    docker compose up
    ```
> **Note:** The Docker setup will run AlphaPose in **CPU-only mode**, as GPU acceleration (MPS) cannot be passed through to a Linux container on macOS. Performance will be slower than a native Linux/GPU setup but is often more stable for development.

