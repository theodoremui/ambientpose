#!/usr/bin/env python3
"""
Download Pretrained Models for AmbientPose

This script downloads pretrained models for AlphaPose pose detection.
Ultralytics and MediaPipe models are downloaded automatically when needed.

Usage:
    python scripts/download_models.py [--all] [--fast-only]
    
Author: AmbientPose Team
Date: 2025-01-16
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List


def download_file(url: str, dest_path: Path, expected_size: int = None, description: str = "", alternative_urls: List[str] = None) -> bool:
    """Download a file with progress indication and fallback URLs."""
    
    urls_to_try = [url] + (alternative_urls or [])
    
    for attempt, current_url in enumerate(urls_to_try):
        try:
            print(f"📥 Downloading {description}...")
            if attempt > 0:
                print(f"   Trying alternative URL {attempt}: {current_url}")
            else:
                print(f"   URL: {current_url}")
            print(f"   Destination: {dest_path}")
            
            # Create directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    print(f"\r   Progress: {percent:3d}% ({block_num * block_size:,} / {total_size:,} bytes)", end="")
            
            # Handle Google Drive URLs specially
            if "drive.google.com" in current_url and "uc?export=download" in current_url:
                import requests
                
                # Download from Google Drive with session handling
                session = requests.Session()
                response = session.get(current_url, stream=True)
                
                # Handle Google Drive virus scan warning
                if 'download_warning' in response.text:
                    for line in response.text.split('\n'):
                        if 'confirm=' in line:
                            confirm_code = line.split('confirm=')[1].split('&')[0]
                            confirmed_url = current_url + f"&confirm={confirm_code}"
                            response = session.get(confirmed_url, stream=True)
                            break
                
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(dest_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    percent = min(100, (downloaded * 100) // total_size)
                                    print(f"\r   Progress: {percent:3d}% ({downloaded:,} / {total_size:,} bytes)", end="")
                    print()  # New line after progress
                else:
                    raise Exception(f"HTTP Error {response.status_code}")
            else:
                # Regular download
                urllib.request.urlretrieve(current_url, dest_path, progress_hook)
                print()  # New line after progress
            
            # Verify file size if provided
            if expected_size and dest_path.stat().st_size != expected_size:
                size_diff = abs(dest_path.stat().st_size - expected_size)
                if size_diff > 1024:  # Allow 1KB difference
                    print(f"⚠️  Warning: Downloaded file size ({dest_path.stat().st_size:,}) doesn't match expected ({expected_size:,})")
            
            print(f"✅ Successfully downloaded: {dest_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download from URL {attempt + 1}: {e}")
            if attempt < len(urls_to_try) - 1:
                print(f"   Trying next URL...")
                continue
    
    print(f"❌ All download attempts failed for {description}")
    return False


def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    if not expected_md5:
        return True
        
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        actual_md5 = hash_md5.hexdigest()
        if actual_md5.lower() == expected_md5.lower():
            print(f"✅ Checksum verified for {file_path.name}")
            return True
        else:
            print(f"❌ Checksum mismatch for {file_path.name}")
            print(f"   Expected: {expected_md5}")
            print(f"   Actual:   {actual_md5}")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not verify checksum for {file_path.name}: {e}")
        return True  # Don't fail on checksum error


def get_model_definitions() -> Dict[str, Dict]:
    """Get definitions of available models."""
    
    # AlphaPose models from various sources
    models = {
        # Fast ResNet50 model (recommended for most use cases)
        "fast_res50_256x192": {
            "url": "https://drive.google.com/uc?export=download&id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn",
            "path": "AlphaPose/pretrained_models/fast_res50_256x192.pth",
            "size": 98431621,  # ~94MB
            "description": "Fast ResNet50 256x192 (Recommended)",
            "md5": "",  # Will skip checksum if empty
            "priority": 1,
            "alternative_urls": [
                "https://github.com/MVIG-SJTU/AlphaPose/releases/download/v0.4.0/fast_res50_256x192.pth",
                "https://onedrive.live.com/download?cid=56B9F9C97F467F17&resid=56B9F9C97F467F17%2115924&authkey=AJsMCMQyKAQMZL4"
            ]
        },
        
        # Alternative models
        "fast_res152_256x192": {
            "url": "https://drive.google.com/uc?export=download&id=1kfyedqyn8exjbbNmYq8XGd2EooQjPtF9",
            "path": "AlphaPose/pretrained_models/fast_res152_256x192.pth", 
            "size": 230444357,  # ~220MB
            "description": "Fast ResNet152 256x192 (Higher accuracy)",
            "md5": "",
            "priority": 2,
            "alternative_urls": []
        },
        
        "fast_res50_384x288": {
            "url": "https://drive.google.com/uc?export=download&id=18jFI_rQZSzHMzzkruv_k6S0-VGpz8hnR",
            "path": "AlphaPose/pretrained_models/fast_res50_384x288.pth",
            "size": 98431621,  # ~94MB
            "description": "Fast ResNet50 384x288 (Higher resolution)",
            "md5": "",
            "priority": 3,
            "alternative_urls": []
        },
        
        # HRNet models (more accurate)
        "hrnet_w32_256x192": {
            "url": "https://drive.google.com/uc?export=download&id=1_wn2ifmoQprBrFoUCTGTse0SKwB9kGgF",
            "path": "AlphaPose/pretrained_models/hrnet_w32_256x192.pth",
            "size": 117798752,  # ~112MB
            "description": "HRNet-W32 256x192 (High accuracy)",
            "md5": "",
            "priority": 4,
            "alternative_urls": []
        }
    }
    
    return models


def download_models(model_names: List[str] = None, fast_only: bool = False) -> bool:
    """Download specified models or all models."""
    
    models = get_model_definitions()
    
    if fast_only:
        # Only download the fast ResNet50 model
        model_names = ["fast_res50_256x192"]
    elif model_names is None:
        # Download all models, sorted by priority
        model_names = sorted(models.keys(), key=lambda x: models[x]["priority"])
    
    print("🚀 AmbientPose Model Downloader")
    print("=" * 50)
    
    # Check what needs to be downloaded
    to_download = []
    for name in model_names:
        if name not in models:
            print(f"❌ Unknown model: {name}")
            continue
            
        model = models[name]
        dest_path = Path(model["path"])
        
        if dest_path.exists():
            file_size = dest_path.stat().st_size
            expected_size = model.get("size", 0)
            
            if expected_size > 0 and abs(file_size - expected_size) > 1024:  # Allow 1KB difference
                print(f"🔄 {dest_path.name} exists but size mismatch, will re-download")
                to_download.append(name)
            else:
                print(f"✅ {dest_path.name} already exists")
        else:
            to_download.append(name)
    
    if not to_download:
        print("\n🎉 All requested models are already downloaded!")
        return True
    
    print(f"\n📦 Need to download {len(to_download)} model(s):")
    total_size = sum(models[name].get("size", 0) for name in to_download)
    print(f"   Total download size: ~{total_size / 1024 / 1024:.1f} MB")
    
    # Download models
    success_count = 0
    for name in to_download:
        model = models[name]
        dest_path = Path(model["path"])
        
        print(f"\n📥 Downloading {name}...")
        success = download_file(
            model["url"], 
            dest_path, 
            model.get("size"),
            model["description"],
            model.get("alternative_urls", [])
        )
        
        if success and model.get("md5"):
            verify_checksum(dest_path, model["md5"])
        
        if success:
            success_count += 1
    
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Success: {success_count}/{len(to_download)}")
    print(f"   ❌ Failed: {len(to_download) - success_count}/{len(to_download)}")
    
    if success_count == len(to_download):
        print("\n🎉 All models downloaded successfully!")
        print("\n📋 Next Steps:")
        print("   1. Run: python scripts/setup_alphapose_env.py")
        print("   2. Test: python cli/detect.py --video data/video/video.avi --backend alphapose")
        return True
    else:
        print(f"\n⚠️  Some downloads failed. Check your internet connection and try again.")
        return False


def list_models():
    """List available models."""
    models = get_model_definitions()
    
    print("📋 Available AlphaPose Models:")
    print("=" * 50)
    
    for name, model in models.items():
        dest_path = Path(model["path"])
        status = "✅ Downloaded" if dest_path.exists() else "⬇️  Available"
        size_mb = model.get("size", 0) / 1024 / 1024
        
        print(f"{status} {name}")
        print(f"   Description: {model['description']}")
        print(f"   Size: ~{size_mb:.1f} MB")
        print(f"   Path: {model['path']}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download pretrained models for AmbientPose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to download (default: download recommended model)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    
    parser.add_argument(
        "--fast-only",
        action="store_true", 
        help="Download only the fast ResNet50 model (recommended)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and their status"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return 0
    
    # Determine which models to download
    if args.all:
        model_names = None  # Download all
    elif args.models:
        model_names = args.models
    else:
        # Default: download the recommended fast model
        model_names = ["fast_res50_256x192"]
        args.fast_only = True
    
    success = download_models(model_names, args.fast_only)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 