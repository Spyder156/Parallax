"""Smoke-test Model A (DINO-injected FPN): instantiate UNTRAINED, run on a bicycle image,
check the output is full-resolution, and PCA->RGB visualize it beside the input.

Untrained: the feature *content* is not meaningful yet (random conv weights); what this
checks is the architecture wiring + that output is sharp full-res (not /14 blocky like raw
DINO). Run:
    docker run --rm --gpus all -v "$PWD:/workspace" \
      -v "$PWD/data/hf_cache:/root/.cache/huggingface" parallax:base \
      python /workspace/scripts/test_model_a.py
"""
import sys, os, glob
sys.path.insert(0, "/workspace")
import numpy as np
import torch
import cv2
from models.model_a import ModelA

DEVICE = "cuda"
IMG_DIR = "/workspace/data/mipnerf360/bicycle/images_4"
OUT = "/workspace/outputs/model_a_fpn"
IN_W, IN_H = 616, 406
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def pca_rgb(feat):
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T.float().cpu().numpy()
    Xc = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    rgb = (Xc @ Vt[:3].T).reshape(H, W, 3)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    model = ModelA(feat_dim=64).to(DEVICE).eval()
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nfroz = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[model] ModelA feat_dim=64 | trainable {ntrain/1e6:.1f}M | frozen(DINO) {nfroz/1e6:.1f}M")

    path = sorted(glob.glob(os.path.join(IMG_DIR, "*.JPG")))[len(glob.glob(os.path.join(IMG_DIR, '*.JPG')))//2]
    img = cv2.resize(cv2.imread(path)[..., ::-1], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    x = torch.tensor(img.copy(), device=DEVICE).float().permute(2, 0, 1)[None] / 255.0
    x = (x - MEAN) / STD

    with torch.no_grad():
        f = model(x)
    print(f"[out] input {tuple(x.shape[-2:])} -> features {tuple(f.shape)} "
          f"(full-res: {f.shape[-2:] == x.shape[-2:]})")

    fr = (pca_rgb(f[0]) * 255).astype(np.uint8)
    pair = np.concatenate([img.astype(np.uint8), fr], 1)
    cv2.imwrite(f"{OUT}/modelA_untrained_{os.path.basename(path).split('.')[0]}.png", pair[..., ::-1])
    print(f"[viz] {OUT}/modelA_untrained_...png  (left=input, right=Model A PCA, UNTRAINED)")


if __name__ == "__main__":
    main()
