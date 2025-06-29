#!/usr/bin/env python3
"""
AlphaPose Environment Setup Script

This script helps set up the environment for AlphaPose integration with AlphaDetect.
It creates the .env file and verifies the AlphaPose installation.

Usage:
    python scripts/setup_alphapose_env.py
    
Author: AlphaDetect Team
Date: 2025-01-16
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_alphapose_directory():
    """Check if AlphaPose directory exists and has the necessary structure."""
    alphapose_dir = Path("AlphaPose")
    
    if not alphapose_dir.exists():
        print("❌ AlphaPose directory not found!")
        print("   Please clone AlphaPose first:")
        print("   git clone https://github.com/MVIG-SJTU/AlphaPose.git")
        return False
    
    required_dirs = ["alphapose", "configs", "detector"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not (alphapose_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ AlphaPose directory is incomplete. Missing: {missing_dirs}")
        return False
    
    print("✅ AlphaPose directory structure looks good!")
    return True


def create_env_file():
    """Create .env file from env.example if it doesn't exist."""
    env_example = Path("env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print("❌ env.example file not found!")
        return False
    
    if env_file.exists():
        print("⚠️  .env file already exists. Skipping creation.")
        return True
    
    try:
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from env.example")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def check_dependencies():
    """Check if required Python dependencies are installed."""
    print("\n🔍 Checking Python dependencies...")
    
    required_packages = [
        "torch", "torchvision", "opencv-python", "numpy", 
        "pillow", "easydict", "yacs", "tensorboardx"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("   Install them with: uv pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required dependencies are installed!")
    return True


def test_alphapose_import():
    """Test if AlphaPose can be imported (either as package or from local directory)."""
    print("\n🧪 Testing AlphaPose import...")
    
    # Add AlphaPose to path if directory exists
    alphapose_path = Path("AlphaPose")
    if alphapose_path.exists():
        sys.path.insert(0, str(alphapose_path))
    
    try:
        # Try importing core AlphaPose modules
        import alphapose
        print("   ✅ alphapose module imported successfully")
        
        try:
            from alphapose.models import builder
            print("   ✅ alphapose.models.builder imported successfully")
        except ImportError as e:
            print(f"   ⚠️  Could not import alphapose.models.builder: {e}")
        
        try:
            from alphapose.utils.config import update_config
            print("   ✅ alphapose.utils.config imported successfully")
        except ImportError as e:
            print(f"   ⚠️  Could not import alphapose.utils.config: {e}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Failed to import AlphaPose: {e}")
        print("   This is expected if AlphaPose package is not installed.")
        print("   The CLI will still work using the local AlphaPose directory.")
        return False


def test_cli_backend():
    """Test if the CLI can detect AlphaPose backend."""
    print("\n🧪 Testing CLI backend detection...")
    
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "from cli.detect import ALPHAPOSE_AVAILABLE; print('AlphaPose available:', ALPHAPOSE_AVAILABLE)"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print(f"   ✅ CLI test result: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ CLI test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ CLI test error: {e}")
        return False


def setup_cuda_environment():
    """Set up CUDA environment variables if CUDA is available."""
    print("\n🔧 Checking CUDA environment...")
    
    # Check for CUDA installation
    cuda_paths = [
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
        Path("C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA"),
    ]
    
    cuda_found = False
    for cuda_base in cuda_paths:
        if cuda_base.exists():
            # Find the latest CUDA version
            cuda_versions = [d for d in cuda_base.iterdir() if d.is_dir() and d.name.startswith("v")]
            if cuda_versions:
                latest_cuda = sorted(cuda_versions)[-1]
                print(f"   ✅ Found CUDA installation: {latest_cuda}")
                
                # Update .env file with correct CUDA path
                env_file = Path(".env")
                if env_file.exists():
                    content = env_file.read_text()
                    content = content.replace(
                        "CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
                        f"CUDA_HOME={latest_cuda}"
                    )
                    content = content.replace(
                        "CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
                        f"CUDA_PATH={latest_cuda}"
                    )
                    env_file.write_text(content)
                    print(f"   ✅ Updated .env with CUDA path: {latest_cuda}")
                
                cuda_found = True
                break
    
    if not cuda_found:
        print("   ⚠️  CUDA not found. CPU-only mode will be used.")
        # Update .env file for CPU-only
        env_file = Path(".env")
        if env_file.exists():
            content = env_file.read_text()
            content = content.replace("ALPHAPOSE_DEVICE=cuda:0", "ALPHAPOSE_DEVICE=cpu")
            content = content.replace("CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=-1")
            env_file.write_text(content)
            print("   ✅ Updated .env for CPU-only mode")
    
    return cuda_found


def main():
    """Main setup function."""
    print("🚀 AlphaPose Environment Setup for AlphaDetect")
    print("=" * 50)
    
    success = True
    
    # Step 1: Check AlphaPose directory
    if not check_alphapose_directory():
        success = False
    
    # Step 2: Create .env file
    if not create_env_file():
        success = False
    
    # Step 3: Check dependencies
    if not check_dependencies():
        success = False
    
    # Step 4: Set up CUDA environment
    cuda_available = setup_cuda_environment()
    
    # Step 5: Test AlphaPose import
    alphapose_import_success = test_alphapose_import()
    
    # Step 6: Test CLI backend detection
    cli_test_success = test_cli_backend()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"   AlphaPose Directory: {'✅' if check_alphapose_directory() else '❌'}")
    print(f"   .env File: {'✅' if Path('.env').exists() else '❌'}")
    print(f"   Dependencies: {'✅' if check_dependencies() else '❌'}")
    print(f"   CUDA Available: {'✅' if cuda_available else '❌'}")
    print(f"   AlphaPose Import: {'✅' if alphapose_import_success else '⚠️'}")
    print(f"   CLI Backend Test: {'✅' if cli_test_success else '❌'}")
    
    if success and cli_test_success:
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Test the CLI with AlphaPose backend:")
        print("   python cli/detect.py --help")
        print("2. Try pose detection:")
        print("   python cli/detect.py --image-dir path/to/images --backend alphapose")
        print("3. Check available backends:")
        print("   python cli/detect.py --image-dir path/to/images --backend auto")
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("   The CLI may still work with MediaPipe or Ultralytics backends.")
        print("   Check the issues above and install missing dependencies.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 