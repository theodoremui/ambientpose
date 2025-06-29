#!/usr/bin/env python3
"""
AlphaPose Installation Helper

This script tries to install AlphaPose correctly after the main environment is set up.
Due to AlphaPose's complex build-time dependencies, it requires special handling.

Usage:
    python scripts/install_alphapose.py

Author: AmbientPose Team
Date: 2025-06-22
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr.strip()}")
        return False


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    print("🔍 Checking NVIDIA GPU...")
    
    # Check GPU using wmic
    try:
        result = subprocess.run("wmic path win32_VideoController get name", 
                               shell=True, check=True, capture_output=True, text=True)
        if "NVIDIA" in result.stdout:
            gpu_info = [line.strip() for line in result.stdout.split('\n') if 'NVIDIA' in line]
            for gpu in gpu_info:
                print(f"✅ Found GPU: {gpu}")
            return True
        else:
            print("❌ No NVIDIA GPU found")
            return False
    except subprocess.CalledProcessError:
        print("❌ Could not check GPU information")
        return False


def check_cuda_installation():
    """Check if CUDA is properly installed."""
    print("🔍 Checking CUDA installation...")
    
    # Check nvcc
    try:
        result = subprocess.run("nvcc --version", shell=True, check=True, capture_output=True, text=True)
        print(f"✅ CUDA Compiler: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("❌ CUDA not installed or not in PATH")
        return False


def check_visual_studio():
    """Check Visual Studio build tools."""
    print("🔍 Checking Visual Studio Build Tools...")
    
    # Check for Visual Studio installations
    vs_paths = [
        "C:\\Program Files\\Microsoft Visual Studio\\2022",
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019",
        "C:\\Program Files\\Microsoft Visual Studio\\2019",
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017"
    ]
    
    for vs_path in vs_paths:
        if Path(vs_path).exists():
            print(f"✅ Found Visual Studio at: {vs_path}")
            return True
    
    # Check for standalone build tools
    build_tools_paths = [
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe"
    ]
    
    for tool_path in build_tools_paths:
        if Path(tool_path).exists():
            print(f"✅ Found VS Build Tools: {tool_path}")
            return True
    
    print("❌ Visual Studio Build Tools not found")
    return False


def fix_uv_environment():
    """Fix UV environment by installing missing packages."""
    print("\n🔧 Fixing UV environment...")
    
    packages_to_install = [
        "pip",
        "setuptools", 
        "wheel",
        "cython",
        "pytest-runner"
    ]
    
    for package in packages_to_install:
        if not run_command(f"uv pip install {package}", f"Installing {package}"):
            return False
    
    return True


def install_precompiled_alternatives():
    """Try to install precompiled alternatives to problematic packages."""
    print("\n📦 Installing precompiled alternatives...")
    
    # Try to install pycocotools with Windows wheels first
    alternatives = [
        ("pycocotools", "Installing pycocotools (Windows-compatible version)"),
        ("cython", "Installing Cython"),
    ]
    
    for package, description in alternatives:
        run_command(f"uv pip install {package}", description)


def install_alphapose_conda_method():
    """Provide instructions for conda-based installation."""
    print("\n🐍 Conda-based Installation Method:")
    print("="*60)
    print("If the main installation fails, try this conda approach:")
    print("")
    print("1. Install Anaconda or Miniconda")
    print("2. Create a new environment:")
    print("   conda create -n alphapose python=3.8 -y")
    print("   conda activate alphapose")
    print("")
    print("3. Install PyTorch with CUDA:")
    print("   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    print("")
    print("4. Install dependencies:")
    print("   pip install cython")
    print("   conda install -c conda-forge libyaml")
    print("")
    print("5. Clone and install AlphaPose:")
    print("   git clone https://github.com/MVIG-SJTU/AlphaPose.git")
    print("   cd AlphaPose")
    print("   python setup.py build develop")


def install_alphapose_methods():
    """Try multiple AlphaPose installation methods."""
    print("\n📦 Attempting AlphaPose installation...")
    
    methods = [
        {
            "name": "Direct Git Installation (no build isolation)",
            "command": 'uv pip install --no-build-isolation "alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose.git"'
        },
        {
            "name": "Manual Clone and Install",
            "commands": [
                "git clone https://github.com/MVIG-SJTU/AlphaPose.git",
                "cd AlphaPose && uv run python -m pip install --no-build-isolation -e ."
            ]
        }
    ]
    
    for method in methods:
        print(f"\n🔧 Trying: {method['name']}")
        
        if 'command' in method:
            if run_command(method['command'], f"Installing AlphaPose via {method['name']}"):
                return True
        elif 'commands' in method:
            success = True
            for cmd in method['commands']:
                if not run_command(cmd, f"Running: {cmd}"):
                    success = False
                    break
            if success:
                return True
        
        print(f"❌ {method['name']} failed, trying next method...")
    
    return False


def verify_installation():
    """Verify that AlphaPose was installed correctly."""
    print("\n🔍 Verifying AlphaPose installation...")
    
    try:
        result = subprocess.run('uv run python -c "import alphapose; print(\'AlphaPose imported successfully\')"', 
                              shell=True, check=True, capture_output=True, text=True)
        print("✅ AlphaPose installed successfully!")
        print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ AlphaPose installation verification failed")
        print(f"Error: {e.stderr.strip()}")
        return False


def print_cuda_installation_guide():
    """Print CUDA installation instructions."""
    print("\n" + "="*60)
    print("🎯 CUDA INSTALLATION REQUIRED")
    print("="*60)
    print("")
    print("Your system has an NVIDIA GPU but CUDA is not installed.")
    print("Please install CUDA before proceeding:")
    print("")
    print("1. Download CUDA 12.6 (recommended):")
    print("   https://developer.nvidia.com/cuda-downloads")
    print("")
    print("2. Select: Windows → x86_64 → 11 → exe (network)")
    print("")
    print("3. Run installer as Administrator")
    print("   - Choose 'Custom' installation")
    print("   - Select: CUDA Toolkit, Driver, Visual Studio Integration")
    print("")
    print("4. Restart your computer after installation")
    print("")
    print("5. Verify installation:")
    print("   nvcc --version")
    print("   nvidia-smi")


def print_visual_studio_guide():
    """Print Visual Studio installation instructions."""
    print("\n" + "="*60)
    print("🔧 VISUAL STUDIO BUILD TOOLS REQUIRED") 
    print("="*60)
    print("")
    print("AlphaPose requires Visual C++ build tools for compilation.")
    print("")
    print("Option 1 - Visual Studio Community (Recommended):")
    print("1. Download from: https://visualstudio.microsoft.com/vs/community/")
    print("2. During installation, select:")
    print("   ✅ Desktop development with C++")
    print("   ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools")
    print("   ✅ Windows 11 SDK")
    print("")
    print("Option 2 - Build Tools Only:")
    print("1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("2. Install C++ build tools workload")
    print("")
    print("After installation, restart and try again.")


def print_alternative_solutions():
    """Print alternative installation methods."""
    print("\n" + "="*60)
    print("🔧 ALTERNATIVE SOLUTIONS")
    print("="*60)
    
    print("\n📋 Solution 1: Use Pre-compiled Wheels")
    print("Some users have success with these:")
    print("  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("  uv pip install 'halpecocotools @ https://files.pythonhosted.org/packages/.../halpecocotools-...-win_amd64.whl'")
    
    print("\n📋 Solution 2: Docker Installation (Recommended)")
    print("Use the provided docker-compose.yml:")
    print("  docker-compose up")
    
    print("\n📋 Solution 3: WSL2 (Windows Subsystem for Linux)")  
    print("Install Ubuntu via WSL2 and follow Linux installation:")
    print("  wsl --install Ubuntu")
    print("  # Then follow Linux AlphaPose installation guides")
    
    install_alphapose_conda_method()


def main():
    """Main installation workflow."""
    print("🚀 AlphaPose Installation Helper")
    print("="*40)
    
    # Check system requirements
    has_nvidia = check_nvidia_gpu()
    has_cuda = check_cuda_installation() if has_nvidia else False
    has_vs = check_visual_studio()
    
    # Provide guidance based on missing components
    if has_nvidia and not has_cuda:
        print_cuda_installation_guide()
        print("\n💡 Please install CUDA first, then run this script again.")
        return 1
    
    if not has_vs:
        print_visual_studio_guide()
        print("\n💡 Please install Visual Studio Build Tools first, then try again.")
        # Don't exit - let them try anyway
    
    # Fix UV environment
    print("\n🔧 Setting up UV environment...")
    if not fix_uv_environment():
        print("❌ Failed to set up UV environment properly")
        return 1
    
    # Install precompiled alternatives
    install_precompiled_alternatives()
    
    # Attempt AlphaPose installation
    if install_alphapose_methods():
        if verify_installation():
            print("\n🎉 AlphaPose installation completed successfully!")
            print("\nYou can now use AlphaPose backend in the CLI:")
            print("  uv run python -m cli.detect --backend alphapose --video your_video.mp4")
            print("\n⚠️  Note: CUDA support requires proper CUDA installation")
            return 0
        else:
            print("\n⚠️  Installation command succeeded but verification failed.")
    
    print("\n❌ Automatic installation failed.")
    print_alternative_solutions()
    
    print("\n💡 Don't worry! The CLI works great with MediaPipe and Ultralytics backends:")
    print("  uv run python -m cli.detect --backend mediapipe --video your_video.mp4")
    print("  uv run python -m cli.detect --backend ultralytics --video your_video.mp4")
    
    return 1


if __name__ == "__main__":
    sys.exit(main()) 