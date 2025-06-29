# This file should be placed at the toplevel of the "AlphaPose" directory
#
# What is this file?  This is an AlphaPose Installation Script for Windows PowerShell

Write-Host "🚀 Starting AlphaPose installation..." -ForegroundColor Green

try {
    # Step 1: Install build dependencies
    Write-Host "📦 Installing build dependencies..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    python -m pip install numpy cython setuptools wheel
    
    # Step 2: Handle pycocotools (Windows compilation issue)
    Write-Host "🔧 Installing pycocotools with Windows wheels..." -ForegroundColor Yellow
    python -m pip install "pycocotools>=2.0.10"
    
    # Step 3: Set up environment
    Write-Host "🌍 Setting up environment..." -ForegroundColor Yellow
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    if (Test-Path $cudaPath) {
        $env:PATH = "$cudaPath\bin;$env:PATH"
        $env:PATH = "$cudaPath\lib\x64;$env:PATH"
        Write-Host "✅ CUDA paths configured"
    } else {
        Write-Warning "⚠️  CUDA not found at $cudaPath"
    }
    
    $env:LC_ALL = "en_US.UTF-8"
    $env:LANG = "en_US.UTF-8"
    
    # Step 4: Modern installation
    Write-Host "⚙️  Installing AlphaPose in development mode..." -ForegroundColor Yellow
    pip install -e . --use-pep517
    
    Write-Host "✅ AlphaPose installation completed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "❌ Installation failed. Trying alternative method..." -ForegroundColor Red
    Write-Host "🔄 Installing from GitHub repository..." -ForegroundColor Yellow
    
    try {
        uv pip install "alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose"
        Write-Host "✅ AlphaPose installed from GitHub!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Both installation methods failed." -ForegroundColor Red
        Write-Host "💡 Consider using MediaPipe alternative: uv pip install mediapipe" -ForegroundColor Cyan
    }
} 