#!/usr/bin/env bash
# Fetch ONE MipNeRF360 indoor scene (images + COLMAP poses) and a converged
# 3DGS .ply for it, for the Stage 0.5 energy-landscape test.
#
# Runs INSIDE the parallax container (keeps host env clean). Mount ./data:
#   docker run --rm --gpus all -v "$PWD/data:/data" -v "$PWD/parallax:/workspace" \
#       parallax:base bash /workspace/scripts/setup_scene.sh room
set -euo pipefail
SCENE="${1:-room}"
DATA=/data

pip install --no-cache-dir nerfbaselines >/dev/null

echo ">> downloading images+poses for scene: ${SCENE}"
nerfbaselines download-dataset "external://mipnerf360/${SCENE}" -o "${DATA}/mipnerf360/${SCENE}"

echo ">> fetching a pretrained gaussian-splatting checkpoint (.ply truth)"
# nerfbaselines stores GS checkpoints per scene; extract the point_cloud .ply.
nerfbaselines download-checkpoint \
    "https://nerfbaselines.github.io/m-gaussian-splatting/mipnerf360/${SCENE}" \
    -o "${DATA}/ckpt/${SCENE}" || \
    echo "!! checkpoint pull failed — fall back to training our own ply (train_gs.sh)"

echo ">> done. contents:"
find "${DATA}/mipnerf360/${SCENE}" -maxdepth 1 -mindepth 1 -printf '   %f\n' | head
ls -R "${DATA}/ckpt/${SCENE}" 2>/dev/null | head || true
