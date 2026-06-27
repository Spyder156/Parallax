# Parallax base image — Blackwell (sm_120) ready.
# Stage 0.5 needs: torch (cu128), the feature-3dgs N-dim rasterizer + simple-knn,
# DINOv2 (Model A, off-the-shelf), and viz. Concerto/Pointcept is layered separately.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Blackwell. +PTX so it still JITs forward-compat if needed.
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV FORCE_CUDA=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git ninja-build build-essential \
    libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip

# --- PyTorch for CUDA 12.8 (Blackwell-capable) ---
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# --- Lightweight deps for Stage 0.5 (no CLIP/LSeg/streamlit cruft) ---
RUN pip install --no-cache-dir \
    numpy plyfile opencv-python-headless pillow tqdm einops matplotlib \
    rerun-sdk transformers timm huggingface_hub scikit-learn

# --- Build the CUDA extensions (feature rasterizer + simple-knn) for sm_120 ---
# Feature payload width is a COMPILE-TIME constant; set it via ARG.
ARG FEATURE_DIM=64
COPY External/feature-3dgs/submodules /opt/feature-3dgs/submodules
RUN sed -i "s/#define NUM_SEMANTIC_CHANNELS .*/#define NUM_SEMANTIC_CHANNELS ${FEATURE_DIM}/" \
    /opt/feature-3dgs/submodules/diff-gaussian-rasterization-feature/cuda_rasterizer/config.h \
    && pip install --no-cache-dir --no-build-isolation /opt/feature-3dgs/submodules/diff-gaussian-rasterization-feature \
    && pip install --no-cache-dir --no-build-isolation /opt/feature-3dgs/submodules/simple-knn

WORKDIR /workspace
