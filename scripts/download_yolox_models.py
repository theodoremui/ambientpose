#!/usr/bin/env python3
"""
Download YOLOX pretrained models for AlphaPose

This script downloads the necessary YOLOX model weights for use with AlphaPose.
"""

import os
import sys
import urllib.request
from pathlib import Path
from loguru import logger

# YOLOX model download URLs
YOLOX_MODELS = {
    'yolox_nano.pth': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth',
    'yolox_tiny.pth': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth', 
    'yolox_s.pth': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth',
    'yolox_m.pth': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth',
    'yolox_l.pth': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth',
    'yolox_x.pth': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth',
    'yolox_darknet.pth': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth'
}

def download_file(url: str, filepath: Path) -> bool:
    """Download a file from URL to filepath with progress."""
    try:
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f'\rDownloading {filepath.name}: {percent}%', end='', flush=True)
        
        urllib.request.urlretrieve(url, filepath, reporthook=show_progress)
        print()  # New line after progress
        logger.success(f"Downloaded: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def main():
    """Download YOLOX models for AlphaPose."""
    
    # Get script directory and AlphaPose path
    script_dir = Path(__file__).parent
    alphapose_dir = script_dir.parent / "AlphaPose"
    
    if not alphapose_dir.exists():
        logger.error(f"AlphaPose directory not found: {alphapose_dir}")
        logger.info("Please ensure AlphaPose is installed in the project directory")
        return 1
    
    # Create YOLOX data directory
    yolox_data_dir = alphapose_dir / "detector" / "yolox" / "data"
    yolox_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {yolox_data_dir}")
    
    # Download models
    download_models = ['yolox_x.pth']  # Start with just the one we need
    
    logger.info("Downloading YOLOX pretrained models...")
    
    success_count = 0
    for model_name in download_models:
        model_path = yolox_data_dir / model_name
        
        # Skip if file already exists and is not empty
        if model_path.exists() and model_path.stat().st_size > 0:
            logger.info(f"Skipping {model_name} (already exists)")
            success_count += 1
            continue
        
        url = YOLOX_MODELS[model_name]
        logger.info(f"Downloading {model_name} from {url}")
        
        if download_file(url, model_path):
            success_count += 1
        else:
            # Clean up failed download
            if model_path.exists():
                model_path.unlink()
    
    logger.info(f"Downloaded {success_count}/{len(download_models)} models successfully")
    
    if success_count == len(download_models):
        logger.success("All YOLOX models downloaded successfully!")
        return 0
    else:
        logger.error("Some downloads failed. Check your internet connection and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 