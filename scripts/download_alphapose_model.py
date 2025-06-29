#!/usr/bin/env python3
"""
Download AlphaPose pretrained model from official Google Drive link.
This script downloads the fast_res50_256x192.pth model that AlphaPose needs.
"""

import os
import requests
from pathlib import Path
import sys
from tqdm import tqdm

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive using the file ID."""
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle the confirmation token for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    # Save the file with progress bar
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"✅ Downloaded: {destination}")

def main():
    # Create the pretrained_models directory
    alphapose_dir = Path("AlphaPose")
    pretrained_dir = alphapose_dir / "pretrained_models"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    # Model file details (from official AlphaPose GitHub issue #1162)
    model_info = {
        "fast_res50_256x192.pth": "1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn"
    }
    
    for filename, file_id in model_info.items():
        destination = pretrained_dir / filename
        
        if destination.exists():
            print(f"⏭️  {filename} already exists, skipping...")
            continue
            
        print(f"📥 Downloading {filename}...")
        try:
            download_file_from_google_drive(file_id, destination)
            print(f"✅ Successfully downloaded {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            sys.exit(1)
    
    print("\n🎉 All AlphaPose models downloaded successfully!")
    print("🚀 AlphaPose is now ready to use with pretrained weights!")

if __name__ == "__main__":
    main() 