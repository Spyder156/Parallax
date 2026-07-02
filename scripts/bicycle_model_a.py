"""Flavor (a) prep: Model A = frozen DINOv2 dense features on the bicycle images.

Extracts patch features, PCA->RGB (shared basis across views), and saves the resized
input beside it so we can eyeball whether A's field is spatially structured and
discriminative (the NO_B question) before baking it onto geometry.

Run:
    docker run --rm --gpus all -v "$PWD:/workspace" \
      -v "$PWD/data/hf_cache:/root/.cache/huggingface" parallax:base \
      python /workspace/scripts/bicycle_model_a.py
"""
import os, glob
import numpy as np
import torch
import cv2
from transformers import AutoModel

DEVICE = "cuda"
IMG_DIR = "/workspace/data/mipnerf360/bicycle/images_4"
OUT = "/workspace/outputs/bicycle/model_a"
PATCH = 14
GW, GH = 44, 29                       # patch grid -> input 616x406 (both %14==0)
IN_W, IN_H = GW * PATCH, GH * PATCH
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def load_image(path):
    img = cv2.imread(path)[..., ::-1]                 # BGR->RGB
    img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    t = torch.tensor(img.copy(), device=DEVICE).float().permute(2, 0, 1)[None] / 255.0
    return img, (t - MEAN) / STD


@torch.no_grad()
def dino_features(model, pix):
    out = model(pixel_values=pix).last_hidden_state    # (1, 1+GW*GH, 768)
    patches = out[:, 1:, :]                             # drop CLS
    return patches.reshape(GH, GW, -1)                 # (GH,GW,768)


def pca_rgb(feat_hw3, basis, mean):
    H, W, C = feat_hw3.shape
    X = feat_hw3.reshape(-1, C).cpu().numpy() - mean
    rgb = (X @ basis).reshape(H, W, 3)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    model = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(DEVICE)
    names = sorted(glob.glob(os.path.join(IMG_DIR, "*.JPG")))
    picks = [names[0], names[len(names) // 2], names[-1]]

    imgs, feats = [], []
    for p in picks:
        rgb, pix = load_image(p)
        f = dino_features(model, pix)
        imgs.append(rgb); feats.append(f)
        print(f"[dino] {os.path.basename(p)} -> feat {tuple(f.shape)}")

    # shared PCA basis across all three views
    allX = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats], 0).cpu().numpy()
    mean = allX.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(allX - mean, full_matrices=False)
    basis = Vt[:3].T

    for p, rgb, f in zip(picks, imgs, feats):
        stem = os.path.basename(p).split(".")[0]
        fr = pca_rgb(f, basis, mean)
        fr = cv2.resize((fr * 255).astype(np.uint8), (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        pair = np.concatenate([rgb.astype(np.uint8), fr], 1)   # input | feature
        cv2.imwrite(f"{OUT}/dino_{stem}.png", pair[..., ::-1])
        print(f"[viz] {OUT}/dino_{stem}.png  (left=input, right=DINOv2 PCA)")
    print("[done] Model A (DINOv2) feature field rendered.")


if __name__ == "__main__":
    main()
