FROM ubuntu:20.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
        && apt-get install --no-install-recommends -y \
        python3-pip \
        python3-dev \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        git \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Set up environment variables for CPU-only build
ENV FORCE_CPU=1
ENV CUDA_HOME=""

# Install base requirements first (including numpy for build dependencies)
COPY requirements.txt .
RUN pip3 install --no-cache-dir numpy cython setuptools wheel
RUN pip3 install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Install AlphaPose with CPU-only build
RUN pip3 install --no-build-isolation "alphapose @ git+https://github.com/MVIG-SJTU/AlphaPose" || \
    pip3 install mediapipe

COPY . .