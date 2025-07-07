# OpenPose Setup and Configuration Guide

*Last updated: 2025-01-22*

This guide provides comprehensive instructions for setting up and configuring **OpenPose** as a backend for AmbientPose. OpenPose is Carnegie Mellon University's real-time multi-person pose estimation framework, offering high accuracy and robust detection capabilities.

---

## Table of Contents
- [OpenPose Setup and Configuration Guide](#openpose-setup-and-configuration-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. System Requirements](#2-system-requirements)
  - [3. OpenPose Installation](#3-openpose-installation)
    - [Windows](#windows)
    - [Linux](#linux)
    - [macOS](#macos)
  - [4. Configuration](#4-configuration)
    - [Environment Variables](#environment-variables)
    - [Python Bindings (Recommended)](#python-bindings-recommended)
    - [Binary Execution Mode](#binary-execution-mode)
  - [5. Usage with AmbientPose](#5-usage-with-ambientpose)
    - [Command Line Interface](#command-line-interface)
    - [Configuration Options](#configuration-options)
  - [6. Performance Optimization](#6-performance-optimization)
  - [7. Troubleshooting](#7-troubleshooting)
  - [8. Comparison with Other Backends](#8-comparison-with-other-backends)
  - [9. Advanced Configuration](#9-advanced-configuration)

---

## 1. Overview

OpenPose provides state-of-the-art pose estimation with the following features:

- **High Accuracy**: Superior keypoint detection compared to many alternatives
- **Multi-Person Support**: Simultaneous detection of multiple people
- **Real-time Performance**: Optimized for GPU acceleration
- **BODY_25 Model**: 25-keypoint pose model for detailed body analysis
- **Robust Detection**: Works well in challenging lighting and crowding conditions

**Integration Modes:**
1. **Python Bindings** (Recommended): Direct Python API for optimal performance
2. **Binary Execution**: Subprocess execution when Python bindings unavailable

---

## 2. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, Ubuntu 18.04, macOS 10.15 | Windows 11, Ubuntu 20.04+ |
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | NVIDIA GTX 1060 (6GB VRAM) | NVIDIA RTX 3070+ (8GB+ VRAM) |
| **CUDA** | CUDA 10.2+ | CUDA 11.8+ |
| **cuDNN** | cuDNN 7.6+ | cuDNN 8.0+ |
| **Disk** | 2 GB for models | 5 GB (including development tools) |

**Note**: OpenPose can run on CPU-only systems, but GPU acceleration is strongly recommended for real-time performance.

---

## 3. OpenPose Installation

### Windows

#### Option 1: Pre-built Binaries (Easiest)

1. **Download OpenPose**:
   ```powershell
   # Download from official releases
   wget https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/openpose-1.7.0-binaries-win64-only_cpu-python3.7-flir-3d_recommended.zip
   
   # Extract to desired location
   Expand-Archive openpose-1.7.0-binaries-win64-*.zip -DestinationPath C:\openpose
   ```

2. **Set Environment Variable**:
   ```powershell
   # Set OPENPOSE_HOME permanently
   [System.Environment]::SetEnvironmentVariable("OPENPOSE_HOME", "C:\openpose", "User")
   
   # Or set for current session
   $env:OPENPOSE_HOME = "C:\openpose"
   ```

3. **Verify Installation**:
   ```powershell
   # Check if binary exists
   Test-Path "$env:OPENPOSE_HOME\bin\OpenPoseDemo.exe"
   ```

#### Option 2: Build from Source (Advanced)

1. **Install Dependencies**:
   ```powershell
   # Install Visual Studio 2019/2022 with C++ workload
   # Install CUDA Toolkit 11.8
   # Install cuDNN 8.x
   # Install CMake 3.12+
   ```

2. **Clone and Build**:
   ```bash
   git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
   cd openpose
   git submodule update --init --recursive
   
   mkdir build
   cd build
   cmake .. -DBUILD_PYTHON=ON -DUSE_CUDNN=ON
   cmake --build . --config Release
   ```

### Linux

#### Option 1: Docker (Recommended)

```bash
# Pull OpenPose Docker image
docker pull cwaffles/openpose

# Run with GPU support
docker run --gpus all -it cwaffles/openpose
```

#### Option 2: Build from Source

1. **Install Dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install -y cmake libopencv-dev libgoogle-glog-dev \
       libgflags-dev libhdf5-dev libatlas-base-dev python3-dev

   # Install CUDA and cuDNN
   # Follow NVIDIA's official installation guide
   ```

2. **Build OpenPose**:
   ```bash
   git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
   cd openpose
   git submodule update --init --recursive
   
   mkdir build && cd build
   cmake .. \
       -DBUILD_PYTHON=ON \
       -DUSE_CUDNN=ON \
       -DGPU_MODE=CUDA \
       -DDOWNLOAD_BODY_25_MODEL=ON
   make -j$(nproc)
   ```

3. **Set Environment Variable**:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export OPENPOSE_HOME=/path/to/openpose
   ```

### macOS

**Note**: OpenPose on macOS has limited GPU support. CPU-only mode is recommended.

1. **Install Dependencies**:
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install dependencies
   brew install cmake opencv glog gflags hdf5 python@3.9
   ```

2. **Build OpenPose**:
   ```bash
   git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
   cd openpose
   git submodule update --init --recursive
   
   mkdir build && cd build
   cmake .. \
       -DBUILD_PYTHON=ON \
       -DGPU_MODE=CPU_ONLY \
       -DDOWNLOAD_BODY_25_MODEL=ON
   make -j$(sysctl -n hw.ncpu)
   ```

---

## 4. Configuration

### Environment Variables

OpenPose integration requires the `OPENPOSE_HOME` environment variable:

```bash
# Linux/macOS
export OPENPOSE_HOME=/path/to/openpose

# Windows PowerShell
$env:OPENPOSE_HOME = "C:\path\to\openpose"

# Windows Command Prompt
set OPENPOSE_HOME=C:\path\to\openpose
```

**Verification**:
```bash
# Verify environment variable is set
echo $OPENPOSE_HOME  # Linux/macOS
echo %OPENPOSE_HOME%  # Windows cmd
echo $env:OPENPOSE_HOME  # Windows PowerShell
```

### Python Bindings (Recommended)

For optimal performance, compile OpenPose with Python bindings:

1. **Build Configuration**:
   ```bash
   cmake .. -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=$(which python3)
   ```

2. **Installation Verification**:
   ```python
   import sys
   sys.path.append('/path/to/openpose/build/python')
   import openpose as op
   print("OpenPose Python bindings available!")
   ```

3. **AmbientPose Integration**:
   AmbientPose will automatically detect and use Python bindings when available.

### Binary Execution Mode

If Python bindings are unavailable, AmbientPose falls back to binary execution:

**Required Files**:
- `$OPENPOSE_HOME/bin/OpenPoseDemo.exe` (Windows)
- `$OPENPOSE_HOME/bin/openpose` (Linux/macOS)
- `$OPENPOSE_HOME/models/` (Model files)

**Model Download** (if not included):
```bash
cd $OPENPOSE_HOME
# Download BODY_25 model
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/body_25/pose_iter_584000.caffemodel -P models/pose/body_25/
```

---

## 5. Usage with AmbientPose

### Command Line Interface

#### Basic Usage

```bash
# Use OpenPose backend explicitly
python cli/detect.py --video input.mp4 --backend openpose

# Auto-select best available backend (may choose OpenPose)
python cli/detect.py --video input.mp4 --backend auto

# Process images with OpenPose
python cli/detect.py --image-dir /path/to/images --backend openpose

# Adjust confidence threshold
python cli/detect.py --video input.mp4 --backend openpose --min-confidence 0.3
```

#### Advanced Options

```bash
# Debug mode for troubleshooting
python cli/detect.py --video input.mp4 --backend openpose --debug

# Custom output location
python cli/detect.py --video input.mp4 --backend openpose --output results.json

# Specify output directory
python cli/detect.py --video input.mp4 --backend openpose --output-dir /custom/output
```

### Configuration Options

OpenPose backend supports the following configurations:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_pose` | `BODY_25` | Pose model (BODY_25, COCO, MPI) |
| `net_resolution` | `656x368` | Network input resolution |
| `face` | `False` | Enable face keypoint detection |
| `hand` | `False` | Enable hand keypoint detection |
| `write_json` | `Auto` | JSON output (binary mode only) |

**Custom Configuration Example**:
```python
# For advanced users modifying the OpenPoseDetector class
params = {
    "model_folder": "/path/to/openpose/models",
    "net_resolution": "832x512",  # Higher resolution for better accuracy
    "model_pose": "BODY_25",
    "face": False,
    "hand": False
}
```

---

## 6. Performance Optimization

### GPU Optimization

1. **CUDA Memory Management**:
   ```bash
   # Monitor GPU memory usage
   nvidia-smi
   
   # Set CUDA visible devices if multiple GPUs
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **Batch Processing**: OpenPose processes single frames, but AmbientPose handles batching efficiently.

3. **Resolution Tuning**:
   - **High Accuracy**: `832x512` (slower)
   - **Balanced**: `656x368` (default)
   - **High Speed**: `432x240` (faster, lower accuracy)

### CPU Optimization

For CPU-only systems:

```bash
# Set number of threads
export OMP_NUM_THREADS=$(nproc)

# Lower resolution for faster processing
# (This requires modifying the OpenPoseDetector configuration)
```

### Memory Management

1. **Reduce Image Resolution**: Process at lower resolution when accuracy permits
2. **Batch Size**: Process videos in chunks for large files
3. **Model Caching**: Keep OpenPose instance alive between detections

---

## 7. Troubleshooting

### Common Issues

#### 1. "OPENPOSE_HOME environment variable not set"

**Solution**:
```bash
# Verify environment variable
echo $OPENPOSE_HOME

# Set if missing
export OPENPOSE_HOME=/path/to/openpose
```

#### 2. "No valid OpenPose Python bindings or binary found"

**Causes & Solutions**:
- **Missing Binary**: Ensure `$OPENPOSE_HOME/bin/` contains executable
- **Python Binding Issues**: Rebuild with `DBUILD_PYTHON=ON`
- **Path Issues**: Verify `OPENPOSE_HOME` points to correct directory

#### 3. CUDA Errors

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

#### 4. Model Loading Errors

```bash
# Check model files exist
ls $OPENPOSE_HOME/models/pose/body_25/

# Download missing models
cd $OPENPOSE_HOME
bash scripts/ubuntu/install_deps.sh  # Downloads models
```

#### 5. Python Import Errors

```python
# Test OpenPose Python bindings
import sys
sys.path.append('/path/to/openpose/build/python')
try:
    import openpose as op
    print("✅ OpenPose Python bindings working")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

### Performance Issues

#### Slow Detection

1. **Check GPU Usage**: `nvidia-smi` during processing
2. **Reduce Resolution**: Lower `net_resolution` parameter
3. **CPU Bottleneck**: Ensure sufficient CPU cores for video decoding

#### Memory Issues

1. **GPU Memory**: Reduce batch size or resolution
2. **System Memory**: Process smaller video chunks
3. **Memory Leaks**: Restart detection for very long videos

### Debugging Commands

```bash
# Test OpenPose directly
cd $OPENPOSE_HOME
./bin/OpenPoseDemo.exe --image_dir examples/media --display 0 --render_pose 0

# Verbose AmbientPose logging
python cli/detect.py --video input.mp4 --backend openpose --debug
```

---

## 8. Comparison with Other Backends

| Feature | OpenPose | MediaPipe | Ultralytics | AlphaPose |
|---------|----------|-----------|-------------|-----------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Multi-person** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup Complexity** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **GPU Requirement** | Recommended | Optional | Optional | Recommended |
| **Keypoints** | 25 (BODY_25) | 33 | 17 (COCO) | 17 (COCO) |

**When to Use OpenPose**:
- ✅ Maximum accuracy is required
- ✅ Multi-person scenarios with crowding
- ✅ Research or production applications
- ✅ GPU resources available
- ❌ Quick prototyping (use MediaPipe)
- ❌ Resource-constrained environments

---

## 9. Advanced Configuration

### Custom Model Training

OpenPose supports custom model training for specific use cases:

1. **Data Preparation**: Annotate custom dataset with COCO format
2. **Training Pipeline**: Use OpenPose training scripts
3. **Integration**: Replace default models in `$OPENPOSE_HOME/models/`

### Multi-GPU Setup

For high-throughput scenarios:

```bash
# Distribute processing across GPUs
CUDA_VISIBLE_DEVICES=0 python cli/detect.py --video batch1.mp4 --backend openpose &
CUDA_VISIBLE_DEVICES=1 python cli/detect.py --video batch2.mp4 --backend openpose &
```

### Integration with Other Tools

OpenPose output is compatible with:
- **3D Reconstruction**: Use with stereo camera setups
- **Motion Capture**: Export to BVH or FBX formats
- **Animation**: Import keypoints into Blender or Maya
- **Sports Analysis**: Track athlete movements

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check OpenPose Documentation**: [Official CMU Repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
2. **AmbientPose Issues**: [GitHub Issues](https://github.com/your-repo/ambientpose/issues)
3. **Community Forums**: OpenPose community discussions
4. **Debug Mode**: Always run with `--debug` flag when reporting issues

**When Reporting Issues, Include**:
- Operating system and version
- CUDA/cuDNN versions
- OpenPose version and build configuration
- Complete error messages
- AmbientPose debug output

---

*Happy Pose Detection! 🎯* 